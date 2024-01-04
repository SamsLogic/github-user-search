import pickle
from typing import List
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from config.log import embedder_logger as logger, CustomAdapterUpload
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

class embedder():
    def __init__(self, project, window=5, min_count=3, workers=4, sg=1, epochs=10, seed=42, vector_size=128, save=True, upsert=False):
        self.project = project
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.seed = seed
        self.vector_size = vector_size
        self.save = save
        self.model = FastText(window=self.window, min_count=self.min_count, workers=self.workers, sg=self.sg, seed=self.seed, vector_size=self.vector_size)
        self.tf_idf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.8, stop_words='english')
        self.upsert = upsert
        self.lemmatizer = WordNetLemmatizer()
        self.embedder_logger = CustomAdapterUpload(logger.getChild('old_embedder'), {'project': project})
    
    def preprocess_text(self, text: str) -> str:
        words = word_tokenize(text)
        words = [word.lower() for word in words]
        words = [re.sub(r'[^a-zA-Z]', '', self.lemmatizer.lemmatize(word)) for word in words]
        words = [word for word in words if word.isalpha()]
        return words

    def embed(self, data: pd.DataFrame, col: str) -> list:
        repo_embeds = []
        for ind, sents in enumerate(data.groupby(by=['user_repo'])[col]):
            self.embedder_logger.info('Preprocessing text data')
            sents = [self.preprocess_text(sent) for sent in sents]

            self.embedder_logger.info('Building FastText vocab')
            self.model.build_vocab(sents)

            self.embedder_logger.info('Training FastText model')
            self.model.train(sents, total_examples=len(sents), epochs=self.epochs)

            self.embedder_logger.info('Training TF-IDF model')
            self.tf_idf_matrix = self.tf_idf.fit_transform([' '.join(sent) for sent in sents])

            doc_embeds = []
            for doc_ind, sent in enumerate(sents):
                word_embeds = []
                for word in sent:
                    word_embeds.append(self.softmax(self.model.wv[word]) * self.tf_idf_matrix[doc_ind, self.tf_idf.vocabulary_[word]])
                doc_embeds.append(self.softmax(np.sum(word_embeds, axis=0)))
        
            with open(f'models/all_embeds_{self.project}.out', 'ab') as f:
                np.savetxt(f, np.array(doc_embeds))
            repo_embeds.append(np.sum(doc_embeds, axis=0)/len(sents))
            self.embedder_logger.info(f'Finished embedding {ind+1} of {len(data.groupby(by=["user_repo"]))}')
        
        if self.save:
            self.embedder_logger.info('Saving embeddings')
            with open(f'models/repo_embeds_{self.project}.out', 'wb') as f:
                np.savetxt(f, np.array(repo_embeds))
            
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def embed_query(self, query):
        query = self.preprocess_text(query)
        words_embed = []
        for word in query:
            words_embed.append(self.model.wv[word])

        query_embed = np.mean(words_embed, axis=0)
        return query_embed