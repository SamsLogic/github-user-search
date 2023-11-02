import openai
from openai.embeddings_utils import cosine_similarity
import backoff
import tiktoken
from config.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
encoder = tiktoken.get_encoding("cl100k_base")

class LLM():
    def __init__(self, project, execution_id, model='gpt-3.5-turbo', temperature=0.5, max_tokens=500, n=1, stop=None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.total_cost = 0
        self.execution_id = execution_id
        self.project = project

    def prepare_messages(self, system_prompt, user_prompt):
        self.messages = [{'role':'system','content':system_prompt}, {'role':'user', 'content':user_prompt}]

    @backoff.on_exception(backoff.expo, openai.error.Timeout)
    def get_openai_response(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
            n=self.n,
            timeout=60
        )
        return response

    def get_explaination(self, text_df, input_query, filenames):
        repo_name = text_df['repo'].unique()[0]

        text_df['data'] = text_df['text_data'].fillna('') + text_df['code'].fillna('')
        text_df['data'] = text_df['data'].apply(lambda x: x[:1000])
        text_chunks = text_df.groupby(['filename'])[['repo', 'filename','data']].apply(lambda x: "File name: " + x['filename'].unique()[0]+ '\n' + '\n'.join(x['data']))

        text_chunks = '\n\n'.join(text_chunks.values.tolist())
        system_prompt = "You are a reviewer whose task is to review github repo and provide your reasoning as per the user request"

        user_prompt = f"""
        You are given some text from a github repo along with the repo name and the filename in which the piece of text is.

        Github repo details:
        Repo name - {repo_name}
        Files in repo - {filenames}
        some code/text snippets from the repo:
        {text_chunks}

        Input Query:
        {input_query}

        Based on the provided information provide a 3 line explaination as to why the given repo and its contents align with the provided input query."""

        messages = self.prepare_messages(system_prompt, user_prompt)
        self.response = self.get_openai_response()
        return self.get_text()

    def get_text(self):
        return self.response['choices'][0]['message']['content']

    def get_usage_cost(self):
        cost = (0.0015/1000)*self.response['usage']['prompt_tokens'] + (0.002/1000)*self.response['usage']['completion_tokens']
        self.total_cost += cost
        return cost

    def get_embedding_cost(self, input_text):
        cost = sum([len(encoder.encode(text)) for text in input_text])*(0.0001/1000)
        self.total_cost += cost
        return cost