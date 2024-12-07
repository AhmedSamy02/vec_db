import os
import gzip
from typing import Annotated
import numpy as np
from sklearn.cluster import KMeans
import joblib
from ivfpq import IVF_PQ

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db_1M.dat", index_file_path="index_1M.dat", new_db=True, db_size=None):
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.nprobe = 30  # Increased nprobe for better retrieval
        # self.ivf_pq = IVF_PQ(nlist=100, m=2, k=128, nprobe=20)  # For 15 Millions
        # self.ivf_pq = IVF_PQ(nlist=100, m=7, k=256, nprobe=20)  # For 10 Millions
        self.ivf_pq = IVF_PQ(nlist=64, m=10, k=256, nprobe=20)  # For 1 Millions
        
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database.")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            self.generate_database(db_size)
        else:
            self.load_index()

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
        self.ivf_pq.fit(vectors)  # Rebuild index with updated settings
        self.save_index()
        
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def retrieve(self, query: np.ndarray, top_k=5):
        query = query.squeeze()
        query /= np.linalg.norm(query)  # Normalize query
        list_ids = self.ivf_pq.search(query, top_k)
        liss_res = []
        for i in list_ids:
            v = self.get_one_row(i)
            liss_res.append((self._cal_score(query, v),i))
        return [x[1] for x in sorted(liss_res, key=lambda x: x[0], reverse=True)[:top_k]]  

    def save_index(self):
        """Save the index to a compressed file."""
        with gzip.open(self.index_path, "wb") as f:
            joblib.dump(
                {
                    "centroids": self.ivf_pq.centroids.astype(np.float16), 
                    "posting_lists": {
                        k: np.array(v, dtype=np.uint32)  # Store as compact arrays
                        for k, v in self.ivf_pq.posting_lists.items()
                    },
                    "subquantizers": [
                        {"centroids": sq.cluster_centers_.astype(np.float16)}  # Reduced precision
                        for sq in self.ivf_pq.subquantizers
                    ],
                    "quantized_data": {
                        k: [np.array(q, dtype=np.uint8) for q in v]  # Compact subquantizer indices
                        for k, v in self.ivf_pq.quantized_data.items()
                    },
                },
                f,
                compress=9,  # Maximum compression
            )

    def load_index(self):
        with gzip.open(self.index_path, "rb") as f:
            try:
                data = joblib.load(f)
                self.ivf_pq.centroids = data["centroids"].astype(np.float16)  # Restore precision
                self.ivf_pq.posting_lists = {
                    k: list(v) for k, v in data["posting_lists"].items()
                }
                self.ivf_pq.subquantizers = []
                for sq in data["subquantizers"]:
                    subquantizer = KMeans(n_clusters=self.ivf_pq.k)
                    subquantizer.cluster_centers_ = sq["centroids"].astype(np.float16)
                    self.ivf_pq.subquantizers.append(subquantizer)
                self.ivf_pq.quantized_data = {
                    k: [np.array(q, dtype=np.uint8) for q in v]
                    for k, v in data["quantized_data"].items()
                }
            except Exception as e:
                print(f"Error loading index: {e}. Rebuilding index...")
                self._build_index()
