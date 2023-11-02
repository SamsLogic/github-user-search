import os
import numpy as np
import pinecone
from config.log import similarity_logger as logger, CustomAdapter
from config.config import PINECONE_INDEX, PINECONE_KEY, EMBEDDINGS_MODEL
from scripts.embeddings import embedder
from scripts.old_embeddings import embedder as old_embedder

pinecone.init(api_key=PINECONE_KEY, environment="gcp-starter")

class getSimilar():
    def __init__(self, project, execution_id, input_query, save=False):
        if f'tfidf_{project}.pkl' in os.listdir('models'):
            self.old = True
        else:
            self.old = False
        if self.old:
            self.embedder = old_embedder(save=save, project=project)
        else:
            self.embedder = embedder(save=save, project=project)
        self.project = project
        self.input_query = input_query
        self.execution_id = execution_id
        self.index = pinecone.Index(PINECONE_INDEX)
        self.similar_logger = CustomAdapter(logger.getChild('getSimilar'), {'project': project, 'executionId': execution_id})
        self.similar_logger.info(f'Embedding query')
        self.query_embed = self.embedder.embed_query(input_query)

    def get_top_n_pinecone(self, top_n = 5, filters = {}):
        text_score = self.index.query(queries=self.query_embed, top_k=top_n, filter=filters)
        return text_score['matches']

    def embed_query(self, input_query):
        return self.embedder.embed_query(input_query)

    def get_top_n_local(self, embed_type='repo', top_n=5, indices=None, embeds=None):
        self.similar_logger.info(f'Loading embeddings')
        if indices is None:
            if embed_type != 'repo':
                raise Exception("Need to pass indices for code embeddings")
            else:
                embeds = np.loadtxt(f'models/{embed_type}_embeds_{self.project}.out')
        else:
            if embeds is None:
                embeds = np.loadtxt(f'models/{embed_type}_embeds_{self.project}.out', skiprows=indices[0], max_rows=indices[-1]-indices[0]+1)
                if len(indices) > 1:
                    indices = [ind - indices[0] for ind in indices]
                    embeds = embeds[indices]
                else:
                    embeds = np.expand_dims(embeds, axis=0)
            
        self.similar_logger.info(f'Getting scores')
        scores = np.dot(embeds, self.query_embed.T)

        self.similar_logger.info(f'Getting top n score args')
        top_n_s = scores.argsort()[-top_n:][::-1]
        top_n_score = scores[scores.argsort()[-top_n:][::-1]]
        return top_n_s, top_n_score