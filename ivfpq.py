from heapq import heappush, heappop
from typing import List
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
import gzip
import os
DIMENSION = 70
BATCH_SIZE = 1000
class IVF_PQ:
    def __init__(self, nlist: int, m: int, k: int, nprobe: int,index_file:str,batch_size: int = 2048):
        self.nlist = nlist
        self.m = m
        self.k = k
        self.nprobe = nprobe 
        self.batch_size = batch_size  
        self.index_path = index_file
        self.base_dir = os.path.join(os.getcwd(), self.index_path)
    def fit(self, vectors: np.ndarray) -> None:
        n, d = vectors.shape
        assert d % self.m == 0

        mbkmeans = MiniBatchKMeans(n_clusters=self.nlist, batch_size=self.batch_size, init='k-means++', max_iter=500, random_state=42)
        self.centroids = mbkmeans.fit(vectors).cluster_centers_
        assignments = mbkmeans.predict(vectors)

        self.posting_lists = {i: [] for i in range(self.nlist)}
        for i, label in enumerate(assignments):
            self.posting_lists[label].append(i)

        d_sub = d // self.m
        for m in range(self.m):
            subvector_data = vectors[:, m * d_sub:(m + 1) * d_sub]
            sub_kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.batch_size, init='k-means++', max_iter=500, random_state=42)
            subquantizer = sub_kmeans.fit(subvector_data)
            self.subquantizers.append(subquantizer)

            for cluster_id, indices in self.posting_lists.items():
                if cluster_id not in self.quantized_data:
                    self.quantized_data[cluster_id] = []
                subvector_cluster = subvector_data[indices]
                quantized_indices = subquantizer.predict(subvector_cluster)
                self.quantized_data[cluster_id].append(quantized_indices)

# gzip and joblib

    def load_centroids(self):
        centroids_file = os.path.join(self.index_path, "centroids.csv")
        with gzip.open(centroids_file, 'rb') as f:
            return joblib.load(f)["centroids"]

    def load_posting_list(self, cluster_id):
        posting_list_file = os.path.join(self.index_path, f"posting_list_{cluster_id}.csv")
        with gzip.open(posting_list_file, 'rb') as f:
            return joblib.load(f)

    def load_subquantizer(self):
        subquantizer_file = os.path.join(self.index_path, "subquantizers_centroids.csv")
        with gzip.open(subquantizer_file, 'rb') as f:
            subquantizers_centroids = joblib.load(f)
        return [sq["centroids"] for sq in subquantizers_centroids]

    def load_quantized_data(self, cluster_id):
        quantized_data_file = os.path.join(self.index_path, f"quantized_data_{cluster_id}.csv")
        with gzip.open(quantized_data_file, 'rb') as f:
            return joblib.load(f)

    def search(self, query: np.ndarray, top_k: int) -> List[int]:
        top_k_nearest = 200 * top_k
        probes = self.nprobe
        if self.index_path == "saved_db_10m":
            top_k_nearest = 100 * top_k
            probes = 10
        elif self.index_path == "saved_db_20m":
            top_k_nearest = 100 * top_k
            probes = 10
        
            
        if query.ndim == 1:
            query = query.reshape(1, -1)

        d_sub = query.shape[1] // self.m

        centroids = self.load_centroids()
        subquantizer = self.load_subquantizer()
        distances_to_centroids = np.linalg.norm(centroids - query, axis=1)
        del centroids
        nearest_clusters = np.argsort(distances_to_centroids)[:probes]
        del distances_to_centroids
        heap = []
        for cluster_id in nearest_clusters:
            posting_list = self.load_posting_list(cluster_id)
            quantized_data = self.load_quantized_data(cluster_id)
            num_points = len(posting_list)
            
            for batch_start in range(0, num_points, self.batch_size):
                batch_indices = posting_list[batch_start:batch_start + self.batch_size]
                reconstructed_vectors = np.zeros((len(batch_indices), query.shape[1]))

                for m in range(self.m):
                    subquantizer_centroids_for_m = subquantizer[m]
                    quantized_indices = quantized_data[m][batch_start:batch_start + self.batch_size]
                    reconstructed_vectors[:, m * d_sub:(m + 1) * d_sub] = subquantizer_centroids_for_m[quantized_indices]

                distances = np.linalg.norm(reconstructed_vectors - query, axis=1)

                for idx, dist in zip(batch_indices, distances):
                    if len(heap) < top_k_nearest:
                        heappush(heap, (-dist, idx))
                    else:
                        if -dist > heap[0][0]:
                            heappop(heap)
                            heappush(heap, (-dist, idx))

        del subquantizer
        del quantized_data
        del posting_list
        results = sorted(heap, key=lambda x: -x[0])
        return [idx for _, idx in results]
    
    
# Memmap
    # def load_centroids(self):
    #     centroids_file = os.path.join(self.index_path, "centroids.csv")
    #     return np.memmap(centroids_file, dtype=np.float16, mode='r', shape=(self.nlist, DIMENSION))
    # def load_posting_list(self, cluster_id):
    #     posting_list_file = os.path.join(self.index_path, f"posting_list_{cluster_id}.csv")
    #     return np.memmap(posting_list_file, dtype=np.uint32, mode='r')
    # def load_subquantizer(self):
    #         subquantizer_centroids = os.path.join(self.index_path, "subquantizer_centroids.csv")
    #         return np.memmap(subquantizer_centroids, dtype='float32', mode='r', shape=(self.m, self.k, DIMENSION//self.m))
    # def load_quantized_data(self, cluster_id):
    #     quantized_data_file = os.path.join(self.index_path, f"quantized_data_{cluster_id}.csv")
    #     quantized_data_memmap = np.memmap(quantized_data_file, dtype=np.uint8, mode='r')
    #     nlist = len(quantized_data_memmap) // self.m  # Approximation, assuming you know the total size
    #     quantized_data_memmap = quantized_data_memmap.reshape(self.m, nlist)
    #     return quantized_data_memmap
    

    # def search(self, query: np.ndarray, top_k: int) -> List[int]:
    #     if query.ndim == 1:
    #         query = query.reshape(1, -1)

    #     d_sub = query.shape[1] // self.m
    #     centroids = self.load_centroids()
    #     subquantizer = self.load_subquantizer()
    #     centroids.reshape(-1, query.shape[1])

    #     distances_to_centroids = np.linalg.norm(centroids - query, axis=1)
    #     del centroids
    #     nearest_clusters = np.argsort(distances_to_centroids)[:self.nprobe]
    #     del distances_to_centroids
    #     heap = []

    #     for cluster_id in nearest_clusters:
    #         posting_list = self.load_posting_list(cluster_id)
    #         quantized_data = self.load_quantized_data(cluster_id)
    #         num_points = len(posting_list)

    #         for batch_start in range(0, num_points, self.batch_size):
    #             batch_indices = posting_list[batch_start:batch_start + self.batch_size]
    #             reconstructed_vectors = np.zeros((len(batch_indices), query.shape[1]))

    #             for m in range(self.m):
    #                 subquantizer_centroids_for_m = subquantizer[m]
    #                 quantized_indices =quantized_data[m][batch_start:batch_start + self.batch_size]
    #                 reconstructed_vectors[:, m * d_sub:(m + 1) * d_sub] = subquantizer_centroids_for_m[quantized_indices]

    #             distances = np.linalg.norm(reconstructed_vectors - query, axis=1)

    #             for idx, dist in zip(batch_indices, distances):
    #                 if len(heap) <  250 * top_k:
    #                     heappush(heap, (-dist, idx))
    #                 else:
    #                     if -dist > heap[0][0]:
    #                         heappop(heap)
    #                         heappush(heap, (-dist, idx))

    #     del subquantizer
    #     del quantized_data
    #     del posting_list
    #     results = sorted(heap, key=lambda x: -x[0])
    #     return [idx for _, idx in results]

