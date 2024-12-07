import heapq
from typing import List
import numpy as np
from sklearn.cluster import MiniBatchKMeans
class IVF_PQ:
    def __init__(self, nlist: int, m: int, k: int, nprobe: int, batch_size: int = 2048):
        self.nlist = nlist  # Number of Voronoi cells (clusters)
        self.m = m  # Number of subspaces for Product Quantization
        self.k = k  # Number of centroids per subspace
        self.nprobe = nprobe  # Number of clusters to probe during search
        self.batch_size = batch_size  # Mini-batch size for KMeans
        self.centroids = None  # Cluster centroids for IVF
        self.posting_lists = None  # Lists of indices for each cluster
        self.subquantizers = []  # PQ subquantizers
        self.quantized_data = {}  # Encoded vectors per posting list

    def fit(self, vectors: np.ndarray) -> None:
        n, d = vectors.shape
        assert d % self.m == 0

        # Step 1: IVF clustering using MiniBatchKMeans
        mbkmeans = MiniBatchKMeans(n_clusters=self.nlist, batch_size=self.batch_size, init='k-means++', max_iter=500, random_state=42)
        self.centroids = mbkmeans.fit(vectors).cluster_centers_
        assignments = mbkmeans.predict(vectors)

        # Step 2: Initialize posting lists
        self.posting_lists = {i: [] for i in range(self.nlist)}
        for i, label in enumerate(assignments):
            self.posting_lists[label].append(i)

        # Step 3: Product Quantization
        d_sub = d // self.m
        for m in range(self.m):
            subvector_data = vectors[:, m * d_sub:(m + 1) * d_sub]
            sub_kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.batch_size, init='k-means++', max_iter=500, random_state=42)
            subquantizer = sub_kmeans.fit(subvector_data)
            self.subquantizers.append(subquantizer)

            # Store quantized indices for each cluster
            for cluster_id, indices in self.posting_lists.items():
                if cluster_id not in self.quantized_data:
                    self.quantized_data[cluster_id] = []
                subvector_cluster = subvector_data[indices]
                quantized_indices = subquantizer.predict(subvector_cluster)
                self.quantized_data[cluster_id].append(quantized_indices)

    def search(self, query: np.ndarray, top_k: int) -> List[int]:
        top_k = top_k * 100
        if query.ndim == 1:
            query = query.reshape(1, -1)

        d_sub = query.shape[1] // self.m

        # Step 1: Find nearest clusters
        distances_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(distances_to_centroids)[:self.nprobe]

        candidates = []
        candidate_scores = []

        # Step 2: Compute distances incrementally to avoid reconstructing all vectors
        for cluster_id in nearest_clusters:
            cluster_data = self.quantized_data[cluster_id]
            cluster_indices = self.posting_lists[cluster_id]

            # Initialize scores for this cluster
            cluster_distances = np.zeros(len(cluster_indices))

            # Incrementally add the distance contributions from each subspace
            for m in range(self.m):
                subquantizer = self.subquantizers[m]
                quantized_indices = cluster_data[m]
                cluster_centers = subquantizer.cluster_centers_[quantized_indices]

                query_segment = query[:, m * d_sub:(m + 1) * d_sub]
                cluster_distances += np.sum((cluster_centers - query_segment) ** 2, axis=1)

            # Update the candidate list
            # use a heap to maintain top-k candidates efficiently
            for i, idx in enumerate(cluster_indices):
                score = -cluster_distances[i]
                if len(candidates) < top_k:
                    heapq.heappush(candidates, idx)
                    heapq.heappush(candidate_scores, score)
                else:
                    if score > candidate_scores[0]:
                        heapq.heappop(candidates)
                        heapq.heappush(candidates, idx)
                        heapq.heappop(candidate_scores)
                        heapq.heappush(candidate_scores, score)

        # Step 3: Select top-k candidates
        candidates = np.array(candidates)
        candidate_scores = np.array(candidate_scores)
        top_k_indices = np.argsort(candidate_scores)[:top_k]

        return candidates[top_k_indices].tolist()
