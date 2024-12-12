import os
import gzip
from typing import Annotated
import numpy as np
import joblib
from ivfpq import IVF_PQ

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db_1M.dat", index_file_path="saved_db_1m", new_db=True, db_size=None)-> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        # self.ivf_pq = IVF_PQ(nlist=100, m=7, k=256, nprobe=10,index_file=index_file_path) # gzip and joblib
        self.ivf_pq = IVF_PQ(nlist=100, m=5, k=256, nprobe=10,index_file=index_file_path) # memmap

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database.")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            self.generate_database(db_size)
        else:
            pass

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize vectors
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)], rebuild_index=True):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        if rebuild_index:
            self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _build_index(self):
        vectors = self.get_all_rows()
        self.ivf_pq.fit(vectors)
        self.save_index()
        
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        query /= np.linalg.norm(query)
        list_ids = self.ivf_pq.search(query, top_k)
        best_ids = []
        for id in list_ids:
            vector= self.get_one_row(id)
            best_ids.append((self._cal_score(query, vector),id))
        return [x[1] for x in sorted(best_ids, key=lambda x: x[0], reverse=True)[:top_k]]     
    
    def save_index(self):
        os.makedirs(self.index_path, exist_ok=True)
        centroids_file = os.path.join(self.index_path, "centroids.csv")
        with gzip.open(centroids_file, "wb") as f:
            joblib.dump(
            {"centroids": self.ivf_pq.centroids.astype(np.float16)},
            f,
            compress=9,
        )
        
        subquantizer_centroids = [
            {"centroids": sq.cluster_centers_.astype(np.float16)} for sq in self.ivf_pq.subquantizers
        ]
        subquantizers_file = os.path.join(self.index_path, "subquantizers_centroids.csv")
        with gzip.open(subquantizers_file, "wb") as f:
            joblib.dump(subquantizer_centroids, f, compress=9)

        for cluster_id, posting_list in self.ivf_pq.posting_lists.items():
            posting_list_file = os.path.join(self.index_path, f"posting_list_{cluster_id}.csv")
            with gzip.open(posting_list_file, "wb") as f:
                joblib.dump(
                    np.array(posting_list, dtype=np.uint32),
                    f,
                    compress=9,
                )

        for cluster_id, quantized_data in self.ivf_pq.quantized_data.items():
            quantized_data_file = os.path.join(self.index_path, f"quantized_data_{cluster_id}.csv")
            with gzip.open(quantized_data_file, "wb") as f:
                joblib.dump(
                    [np.array(q, dtype=np.uint8) for q in quantized_data],
                    f,
                    compress=9,
                )
