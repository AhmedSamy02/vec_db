import numpy as np
from sklearn.cluster import KMeans
from typing import List

class IVFPQ:
    def __init__(self, nlist: int, m: int, nbits: int):
        """
        :param nlist: Number of Voronoi cells (partitions)
        :param m: Number of sub-vectors for Product Quantization
        :param nbits: Number of bits for encoding (affects the number of centroids in each subspace)
        """
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.centroids = None
        self.subspace_centroids = []
        self.assignments = []
        self.vectors = None  # Store original data for retrieval
    
    def fit(self, vectors: np.ndarray) -> None:
        """
        Fit the IVF-PQ index with the given data.
        :param vectors: The dataset to index (NxD numpy array)
        """
        n, d = vectors.shape
        if d % self.m != 0:
            raise ValueError("The vector dimensionality must be divisible by m.")

        self.vectors = vectors  # Store original vectors
        # Step 1: Cluster the dataset into nlist partitions
        kmeans = KMeans(n_clusters=self.nlist, random_state=42)
        self.assignments = kmeans.fit_predict(vectors)
        self.centroids = kmeans.cluster_centers_

        # Step 2: Subdivide each vector into m sub-vectors
        subvector_dim = d // self.m
        for i in range(self.m):
            subvectors = vectors[:, i * subvector_dim:(i + 1) * subvector_dim]

            # Cluster each subvector space
            kmeans_sub = KMeans(n_clusters=2 ** self.nbits, random_state=42)
            kmeans_sub.fit(subvectors)
            self.subspace_centroids.append(kmeans_sub.cluster_centers_)
    
    def encode(self, vectors: np.ndarray) -> List[np.ndarray]:
        """
        Encode vectors into compressed representations using the PQ centroids.
        :param vectors: The vectors to encode (NxD numpy array)
        :return: Encoded vectors
        """
        encoded_vectors = []
        d = vectors.shape[1]
        subvector_dim = d // self.m

        for i in range(self.m):
            subvectors = vectors[:, i * subvector_dim:(i + 1) * subvector_dim]
            centroid = self.subspace_centroids[i]
            encoded = np.argmin(
                np.linalg.norm(subvectors[:, None, :] - centroid[None, :, :], axis=2), axis=1
            )
            encoded_vectors.append(encoded)
        
        return np.array(encoded_vectors).T

    def search(self, query: np.ndarray, top_k: int) -> List[np.ndarray]:
        """
        Search for the nearest neighbors using IVF-PQ.
        :param query: The query vector (1xD numpy array)
        :param top_k: Number of top results to return
        :return: List of top-k nearest neighbors
        """
        d = query.shape[1]
        subvector_dim = d // self.m

        # Step 1: Assign query to the nearest centroid
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_centroid_idx = np.argmin(centroid_distances)

        # Step 2: Retrieve candidate indices from the nearest partition
        candidates_idx = np.where(self.assignments == nearest_centroid_idx)[0]
        candidates = self.vectors[candidates_idx]  # Fetch the actual vectors for candidates

        # Step 3: Compute the PQ distance for candidates
        pq_distances = []
        for candidate_idx in range(len(candidates)):
            candidate = candidates[candidate_idx]
            distance = 0
            for i in range(self.m):
                query_sub = query[0, i * subvector_dim:(i + 1) * subvector_dim]
                centroid = self.subspace_centroids[i]
                encoded = np.argmin(np.linalg.norm(centroid - query_sub, axis=1))
                distance += np.linalg.norm(candidate[i * subvector_dim:(i + 1) * subvector_dim] - centroid[encoded])
            pq_distances.append((candidate_idx, distance))

        # Sort candidates by distance and return top_k
        pq_distances.sort(key=lambda x: x[1])
        top_k_results = [candidates[idx[0]] for idx in pq_distances[:top_k]]
        return top_k_results