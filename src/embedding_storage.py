from typing import Iterable
import faiss
import numpy as np


class EmbeddingStorage:
    def __init__(self, n_dim: int = 256) -> None:
        self.n_dim = n_dim
        self.index = faiss.index_factory(n_dim, 
                                         'Flat', 
                                         faiss.METRIC_INNER_PRODUCT)

    def __len__(self):
        return self.index.ntotal    
    
    def add(self, embeddings: np.ndarray) -> None:
        embeddings = self._prepare_vectors(embeddings)
        self.index.add(embeddings)
    
    def filtered_search(self, query: np.ndarray, k: int, threshold: float) -> Iterable[np.ndarray]:
        query = self._prepare_vectors(query)
        dists, indices = self.index.search(query, k)
        
        more_than_thrs = dists[:, 0] >= threshold
        filtered_dists = dists[more_than_thrs]
        filtered_indices = indices[more_than_thrs]
        
        return filtered_dists, filtered_indices
    
    def _prepare_vectors(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        if len(vectors.shape) > 2 and vectors.shape[1] != self.n_dim:
            vectors = np.concatenate(vectors, axis=0)
        faiss.normalize_L2(vectors)
        return vectors
    
    def remove(self, indices_to_remove: np.ndarray) -> None:
        self.index.remove_ids(indices_to_remove.astype(np.int64))


if __name__ == '__main__':
    es = EmbeddingStorage()
