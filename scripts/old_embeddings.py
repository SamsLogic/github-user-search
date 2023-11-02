import pickle
from typing import List, Optional
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from config.log import embedder_logger as logger, CustomAdapterUpload

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
        self.embedder_logger = CustomAdapterUpload(logger.getChild('old_embedder'), {'project': project})
    
    def get_list_of_words(self, series: list) -> List[list]:
        sents = []
        for sent in series:
            sents.extend(sent.split(' '))
        return sents        

    def embed(self, data: pd.DataFrame, col: str) -> list:
        self.embedder_logger.info('Getting list of words')
        sents = self.get_list_of_words(data[col])

        self.embedder_logger.info('Building FastText vocab')
        self.model.build_vocab(sents)

        self.embedder_logger.info('Training FastText model')
        self.model.train(sents, total_examples=len(sents), epochs=self.epochs)

        self.embedder_logger.info('Training TF-IDF model')
        self.tf_idf.fit(data[col])

        self.embedder_logger.info('Getting IDF weights')
        self.df_idf = pd.DataFrame({'idf_weights':self.tf_idf.idf_, "words":self.tf_idf.get_feature_names_out()})
        if self.save:
            self.embedder_logger.info('Saving models')
            self.model.save(f"models/fasttext_{self.project}.model")
            with open(f"models/tfidf_{self.project}.pkl", "wb") as f:
                pickle.dump(self.tf_idf, f)
            self.df_idf.to_excel(f"models/dfidf_{self.project}.xlsx", index=True)

        for ind, sents in enumerate(data.groupby(by=['user_repo'])[col]):
            sent = self.get_list_of_words(sents[1])[:200]
            words_embed = [self.model.wv[word] for word in sent]
            tf_idf_weights = []
            for word in sent:
                if len(self.df_idf.loc[self.df_idf['words'] == word, 'idf_weights']) != 0:
                    tf_idf_weights.append(self.df_idf.loc[self.df_idf['words'] == word, 'idf_weights'].values[0])
                else:
                    tf_idf_weights.append(0)
            words_embed = np.array(words_embed)
            tf_idf_weights = np.array(tf_idf_weights)
            words_embed = self.softmax(words_embed)*tf_idf_weights[:, None]
            sent_embed = np.sum(words_embed, axis=0)/len(words_embed)
            with open(f'models/repo_embeds_{self.project}.out', 'ab') as f:
                repo_embeds = np.array([sent_embed])
                np.savetxt(f, repo_embeds)
            self.embedder_logger.info(f'Finished embedding {ind+1} of {len(data.groupby(by=["user_repo"]))}')
            
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def load_models(self):
        self.model = FastText.load(f"models/fasttext_{self.project}.model")
        self.tf_idf = pickle.load(open(f"models/tfidf_{self.project}.pkl", "rb"))

    def embed_query(self, query):
        self.load_models()
        query = query.split(' ')
        words_embed = []
        for word in query:
            words_embed.append(self.model.wv[word])

        query_embed = np.sum(words_embed, axis=0)
        return np.array(query_embed)