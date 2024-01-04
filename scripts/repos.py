import pandas as pd
import numpy as np
from scripts.similarity import getSimilar
from scripts.llm import LLM
from config.log import repo_logger as logger, CustomAdapter

class getRepos():
    def __init__(self, query, execution_id, project, top_n=5, stars=0):
        self.repo_logger = CustomAdapter(logger, {'executionId': execution_id, 'project': project})
        self.input_query = query
        self.repo_logger.info(f'Query: {self.input_query}')
        self.top_n = top_n
        self.stars = stars
        self.project = project
        self.execution_id = execution_id
        self.repo_logger.debug('Loading data')
        self.data = pd.read_excel(f'data/{project}_data.xlsx')
        self.languages = set()
        self.embedder = getSimilar(project=project, execution_id=execution_id, input_query=self.input_query, save=False)
        self.llm = LLM(project=project, execution_id=execution_id)
        self.repo_logger.debug('Preprocessing text data')
        self.data['languages'] = self.data['languages'].fillna('')
        for langs in self.data['languages'].unique():
            for lang in langs.split(', '):
                if lang == '':
                    continue
                self.languages.add(lang)
    
    def filter_on_lang(self, data):
        if any([i.lower() in self.languages for i in self.input_query.split(' ')]):
            return data[data['languages'].apply(lambda x: any([i.lower() in x.split(', ') for i in self.input_query.split(' ')]))]
        else:
            return data

    def filter_on_stars(self, data):
        return data[data['stars'] >= self.stars]

    def get_repos(self):
        output = {i:{} for i in range(self.top_n)}
        user_repo_data = self.data.groupby(by=['user_repo']).first().reset_index()
        self.repo_logger.info('Filtering on language')
        user_repo_data = self.filter_on_lang(user_repo_data)
        self.repo_logger.info('Filtering on stars')
        user_repo_data = self.filter_on_stars(user_repo_data)

        if self.data is None:
            raise Exception('No repos found for the given query')

        self.repo_logger.info('Getting top n repo score')
        top_n_repo_args, repo_score = self.embedder.get_top_n_local(top_n=self.top_n, indices=user_repo_data.index.tolist(), embed_type='repo')
        
        self.repo_logger.info('Getting top n repos')
        top_n_user_repos = user_repo_data.iloc[top_n_repo_args].reset_index(drop=True)
        top_n_user_repos['repo_score'] = repo_score
        print(top_n_user_repos)
        output = {ind:{"user":i.split('_')[0], "repo":'_'.join(i.split('_')[1:])} for ind, i in enumerate(top_n_user_repos['user_repo'].tolist())}
        # run for each repo
        for ind, user_repo in enumerate(top_n_user_repos['user_repo']):
            output[ind]['repo_score'] = top_n_user_repos['repo_score'].values[ind]
            repo_data = self.data[self.data['user_repo'] == user_repo]
            repo_data = repo_data.fillna('')
            repo_data['strings'] = repo_data['text_strings'] + repo_data['code_strings']
            self.repo_logger.info(f'Getting top 3 text score for explaintation for {user_repo}')
            repo_comp_args, _ = self.embedder.get_top_n_local(top_n=3, indices=repo_data.index.tolist(), embed_type='all')
            repo_comp_args = repo_data.iloc[repo_comp_args].reset_index(drop=True)
            explaination = self.llm.get_explaination(repo_comp_args, self.input_query, repo_data['filename'].tolist())
            
            self.repo_logger.info(f'Getting explaination for {user_repo}')
            output[ind]['explaination'] = explaination

            repo_data = repo_data[repo_data['code_strings'] != '']
            self.repo_logger.info(f'Getting top 2 code snippets for {user_repo}')
            if len(repo_data) > 0:
                repo_comp_args, snippet_score = self.embedder.get_top_n_local(top_n=2, indices=repo_data.index.tolist(), embed_type='all')
                repo_comp_args = repo_data.iloc[repo_comp_args].reset_index(drop=True)
                repo_comp_args['snippet_score'] = snippet_score
                repo_data_output = {ind:{"filename":i['filename'], "code":i['code'], "score":i['snippet_score']} for ind, i in repo_comp_args[['filename', 'code', 'snippet_score']].iterrows()}
            else:
                repo_data_output = {0:{"filename": "NA", "code":"NA", "score": "NA"}}
            output[ind]['code_snippet'] = repo_data_output
            
        return output



        
        