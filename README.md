# Github Users Screening

Hey, this is a screening project to search/screen top N number of github profiles based on natural language query. At the current state of AI there are multiple ways of solving such problem, by using pretrained embeddings, training a TF_IDF model, training a BERT embeddings model, training a T5 embeddings model, or many more.

In this repo, we will be looking in to the first two approaches mentioned and will compare the results as well.
1. Using pretrained embeddings model
2. Training a FastText + TF_IDF vectorizer model (Inspired by - [NCS Meta](https://ai.meta.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/)) 

Using this code i am trying to answer below mentioned questions
1. Identification of top N number of github repos based on the provided input query
2. Provide explaination for each repo's relevancy
3. Identify code snippets relevant to the input query from the top N repos

Tasks required to be performed:
1. Use Github APIs to extract repos for each and every user. Below are the mentioned data points extracted per user.
    1. Public Repository
    2. Files within the repo (.md, .py, .js, .java)
    3. languages used in the repo
    4. stars on the repo
    5. language used by each file
2. Parse through each code file and extract only relevant blocks of code. Below are blocks extracts:
    1. Function definition
    2. Function call
    3. For loops
3. Generate embeddings of the textual and code components. (This is different for the 2nd approach)
4. Aggregate the generate embeddings using some formula to get repo embeddings
2. Filter over language if mentioned in the natural language query
3. Filter over stars if provided as input
4. Generate embeddings of the input query
5. Get top n repo similar to the input query using the embeddings
6. For each repo:
    1. Get explaination of the repo by getting the top 3 components of the repo be it text or code as well as all the components of the repo - repo name, filename, file contents (top 3)
    2. Get top 2 code snippets similar to the input query using embeddings similarity on the code blocks from the repo.
7. Provide the output in JSON format.
8. Build an API on this

## Demo

To try out the apis run the below curl command

```terminal
curl --location 'http://ec2-54-153-3-47.us-west-1.compute.amazonaws.com/api/v0.1/repo/repo' \
--header 'Content-Type: application/json' \
--header 'Content-Type: application/json' \
--data '{
    "query": "computer vision and deep learning based project",
    "project": "github-users-all"
}'
```

## Using pretrained embeddings model (Default)

### Overview

For this approach, i am using sbert pretrained model - `all-MiniLM-L6-v2` which has been trained on StackExchange dataset as well. To start, i am embedding the textual and code components extracted from the repo files and then aggregrating the embeddings using the l1 distance/Manhatan distance to get the centroid of the embeddings assuming the embeddings are part of a cluster where each repo is a cluster and the repo's collective embedding is the centroid of the text and code component embeddings

### Advantages

1. Much better results
2. Good results without finetuning or training
3. Easy to deploy
4. Low text preprocessing required

### Disadvantages

1. Slow embedding generation on CPUs
2. Less explinabale

### Further steps

1. Adding cross-encoders for reranking of the top n repos
2. Trying out larger models and using GPUs instead of inferencing

## Using FastText + TF_IDF vectorizer model

### Overview

For this approach i am training a FastText model on word embeddings of each component of code and text. These word embeddings give me an embedding array of each word. Then performing TF-IDF on the whole dorpus to identify the weight of each word in the corpus which is used to get a weighted average repo embeddings averaged over the weighted sum of the word embeddings of each sentence. Below is the mathematical representation of the same.

$$ \text{avg}(\text{avg} (\text{wordembeddings} \times \text{tfidfwordweight})) $$

### Advantages

1. Faster embedding generation
2. Much more explainable

### Disadvantages

1. Finding the best approach to get repo embeddings
2. The results are not promosing

## Challenges faced in development

Some of the major challenges are,
1. Extracting selected code blocks from source code for multiple languages
2. Identifying the right approach for repo embeddings
3. Saving the huge corpus of embeddings and fecthin only selected embeddings without overflowing the memory
4. Using TF_IDF and FastText word embeddings together and generating repo embeddings

## Setup

1. First clone this repo-

```terminal
git clone [repo-name]
```

2. Maintin environment variables in the .env file.
```terminal
cd [repo-name]
cd config/
cp .env.example .env
```

3. clone the parser repos to the build directory - [tree-sitter-python](https://github.com/tree-sitter/tree-sitter-python), [tree-sitter-java](https://github.com/tree-sitter/tree-sitter-java), [tree-sitter-javascript](https://github.com/tree-sitter/tree-sitter-javascript)
```terminal
cd build
git clone git@github.com:tree-sitter/tree-sitter-python.git
git clone git@github.com:tree-sitter/tree-sitter-java.git
git clone git@github.com:tree-sitter/tree-sitter-javascript.git
```

### With Docker

1. Build repo image
```bash
docker build -t github-repo:v1 .
```

2. run a container for this image
```
docker run -d -p 80:5000 -v ~/github-user-search/logs:/app/logs --env-file ~/github-user-search/config/.env github:v1
```

3. Your api is up, you can hit the api and get your results.

### Without Docker

1. Create a virtual env
```terminal
pip install virutalenv
python -m virtualenv venv
source venv/bin/activate
```

2. Install dependencies
```terminal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

3. run python app
```terminal
python app.py
```