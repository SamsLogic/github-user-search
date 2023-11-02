import os
from typing import List, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config.log import embedder_logger as logger, CustomAdapterUpload
from config.config import EMBEDDINGS_MODEL

checkpoint = EMBEDDINGS_MODEL

model = SentenceTransformer(checkpoint)

class embedder():
    def __init__(self, project, vector_size=256, save=True, upsert=False, batch_size = 500):
        self.vector_size = vector_size
        self.save = save
        self.upsert = upsert
        self.batch_size = batch_size
        self.project = project
        self.embedder_logger = CustomAdapterUpload(logger.getChild('embedder'), {'project': project})

    def embed(self, data: pd.DataFrame, col: str) -> list:
        if f'repo_embeds_{self.project}.npy' not in os.listdir("models"):
            self.embedder_logger.info('Getting repo embeddings')
            repo_embeds = []
            for ind, sents in enumerate(data.groupby(by=['user_repo'])[col]):
                embeddings = model.encode(sents[1].values, batch_size=self.batch_size)
                with open(f'models/all_embeds_{self.project}.out', 'ab') as f:
                    np.savetxt(f, embeddings)
                repo_embeds.append(np.sum(embeddings, axis=0)/len(sents[1]))
                self.embedder_logger.info(f'Finished embedding {ind+1} of {len(data.groupby(by=["user_repo"]))}')
            
            if self.save:
                self.embedder_logger.info('Saving embeddings')
                with open(f'models/repo_embeds_{self.project}.out', 'wb') as f:
                    np.savetxt(f, repo_embeds)

    def embed_query(self, query: str) -> np.array:
        return model.encode(query)
