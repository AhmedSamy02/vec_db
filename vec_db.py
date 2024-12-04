# from typing import Dict, List, Annotated
# import numpy as np
# import os
# from ivfpq import IVFPQ

# DB_SEED_NUMBER = 42
# ELEMENT_SIZE = np.dtype(np.float32).itemsize
# DIMENSION = 70

# class VecDB:
#     def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
#         self.db_path = database_file_path
#         self.index_path = index_file_path
#         self.ivfpq = IVFPQ(150, 70, 4)
#         self._build_index()

#         # if new_db and db_size:
#         #     rng = np.random.default_rng(DB_SEED_NUMBER)
#         #     temp_data = rng.random((db_size, DIMENSION), dtype=np.float32)
#         #     nlist, nsubvectors, nbits = adjust_parameters(temp_data)
#         # else:
#         #     nlist, nsubvectors, nbits = 20, 8, 6  # Default values

#         # self.ivfpq = IVFPQ(20, 10, 6)

#         if new_db:
#             if db_size is None:
#                 raise ValueError("You need to provide the size of the database")
#             # delete the old DB file if exists
#             if os.path.exists(self.db_path):
#                 os.remove(self.db_path)
#             self.generate_database(db_size)
    
#     def generate_database(self, size: int) -> None:
#         rng = np.random.default_rng(DB_SEED_NUMBER)
#         vectors = rng.random((size, DIMENSION), dtype=np.float32)
#         self._write_vectors_to_file(vectors)
#         self._build_index()

#     def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
#         mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
#         mmap_vectors[:] = vectors[:]
#         mmap_vectors.flush()

#     def _get_num_records(self) -> int:
#         return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

#     def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
#         num_old_records = self._get_num_records()
#         num_new_records = len(rows)
#         full_shape = (num_old_records + num_new_records, DIMENSION)
#         mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
#         mmap_vectors[num_old_records:] = rows
#         mmap_vectors.flush()
#         #TODO: might change to call insert in the index, if you need
#         # self._build_index()

#     def get_one_row(self, row_num: int) -> np.ndarray:
#         # This function is only load one row in memory
#         try:
#             offset = row_num * DIMENSION * ELEMENT_SIZE
#             mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
#             return np.array(mmap_vector[0])
#         except Exception as e:
#             return f"An error occurred: {e}"

#     def get_all_rows(self) -> np.ndarray:
#         # Take care this load all the data in memory
#         num_records = self._get_num_records()
#         vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
#         return np.array(vectors)
    
#     def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
#         # scores = []
#         # num_records = self._get_num_records()
#         # # here we assume that the row number is the ID of each vector
#         # for row_num in range(num_records):
#         #     vector = self.get_one_row(row_num)
#         #     score = self._cal_score(query, vector)
#         #     scores.append((score, row_num))
#         # # here we assume that if two rows have the same score, return the lowest ID
#         # scores = sorted(scores, reverse=True)[:top_k]
#         # return [s[1] for s in scores]
#         candidates = self.ivfpq.search(query, top_k)
#         # Directly fetch the rows based on index
#         return [self.get_all_rows().tolist().index(candidate.tolist()) for candidate in candidates]
    
#     def _cal_score(self, vec1, vec2):
#         dot_product = np.dot(vec1, vec2)
#         norm_vec1 = np.linalg.norm(vec1)
#         norm_vec2 = np.linalg.norm(vec2)
#         cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
#         return cosine_similarity

#     def _build_index(self):
#         # Placeholder for index building logic
#         data = self.get_all_rows()
#         self.ivfpq.fit(data)

from typing import Annotated
import numpy as np
import os
from b import MergedIVFPQ

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        # self.ivfpq = IVFPQ(nlist=150, m=4, nbits=8)
        self.ivfpq =  MergedIVFPQ(d=DIMENSION,nlist=3000, m=35, bits_per_subvector=10)
        # self.ivfpq =  MergedIVFPQ(d=DIMENSION,nlist=10000, m=70, bits_per_subvector=8) 10**4 0.0 0.09

        self._build_index(self.get_all_rows())
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index(vectors)
    
    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)
    
    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index(mmap_vectors)  # Rebuild index after inserting new records
    
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
    def get_batch(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Load a batch of vectors from the database using indices.

        :param start_idx: Starting index (inclusive)
        :param end_idx: Ending index (exclusive)
        :return: A NumPy array containing the batch of vectors
        """
        try:
            num_records = self._get_num_records()
            # Ensure indices are within bounds
            if start_idx < 0 or end_idx > num_records or start_idx >= end_idx:
                raise IndexError("Invalid start or end index for batch retrieval.")

            # Calculate the number of rows to fetch and their byte offset
            num_rows = end_idx - start_idx
            offset = start_idx * DIMENSION * ELEMENT_SIZE

            # Memory-map the batch into memory
            mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                                     shape=(num_rows, DIMENSION), offset=offset)

            # Convert to a standard NumPy array for easier handling
            return np.array(mmap_vectors)
        except Exception as e:
            return f"An error occurred: {e}"
    def retrieve_old(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        print("Retrieving")
        # candidates = self.ivfpq.search(query, top_k)
        # return [self.get_all_rows().tolist().index(candidate.tolist()) for candidate in candidates]
        # query /= np.linalg.norm(query, axis=1, keepdims=True)

        return self.ivfpq.search(query,top_k, nprobe=100)
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def _build_index(self, vectors: np.ndarray) -> None:
        """
        Incrementally builds the IVFPQ index using batches of vectors.
        """
        # vectors = self.get_all_rows()
        # vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        print( "Building index")
        self.ivfpq.train(vectors)
        print("fitting")
        self.ivfpq.encode(vectors)
        print("done")

        # num_records = self._get_num_records()
        # step = max(num_records // 1, 1)
        # for i in range(0, num_records, step):
        #     batch_vectors = self.get_batch(i, min(i + step, num_records))
        #     batch_vectors /= np.linalg.norm(batch_vectors, axis=1, keepdims=True)

        #     print("Training IVFPQ index on batch...")
        #     self.ivfpq.train_batch(batch_vectors)
        #     print("Encoding batch...")
        #     self.ivfpq.encode_batch(batch_vectors)
        #     print("Finished batch.")

