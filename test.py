import numpy as np
from sklearn.cluster import KMeans

class IVFPQ:
    def __init__(self, d, nlist, m, bits_per_subvector):
        self.d = d  # Dimensionality
        self.nlist = nlist  # Number of clusters for IVF
        self.m = m  # Number of subvectors for PQ
        self.k = 2 ** bits_per_subvector  # Number of centroids per subvector (codebook size)
        self.subvector_dim = d // m  # Dimensionality per subvector

        assert d % m == 0, "Dimensionality must be divisible by m."

        # KMeans for IVF (inverted index)
        self.ivf = KMeans(n_clusters=nlist, random_state=42)

        # KMeans for PQ codebooks, one for each subvector
        self.pq_codebooks = [KMeans(n_clusters=self.k, random_state=42) for _ in range(m)]

        # Inverted lists to store encoded vectors
        self.inverted_lists = {i: [] for i in range(nlist)}

    def train(self, X):
        # Step 1: Train IVF on the full dataset
        self.ivf.fit(X)

        # Step 2: Initialize empty lists for cluster-specific subvectors
        cluster_subvectors = {i: [] for i in range(self.nlist)}

        # Step 3: Assign each vector to its nearest cluster
        cluster_ids = self.ivf.predict(X)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_subvectors[cluster_id].append(X[idx])

        # Step 4: Train PQ codebooks on data within each cluster
        for cluster_id, vectors in cluster_subvectors.items():
            if len(vectors) == 0:
                continue
            cluster_data = np.array(vectors)
            for i in range(self.m):
                subvectors = cluster_data[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
                self.pq_codebooks[i].fit(subvectors)

    def encode(self, X):
        # Step 1: Assign vectors to IVF clusters
        cluster_ids = self.ivf.predict(X)

        # Step 2: Encode vectors using PQ codebooks
        pq_codes = np.zeros((X.shape[0], self.m), dtype=np.int32)
        for i in range(self.m):
            subvectors = X[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
            pq_codes[:, i] = self.pq_codebooks[i].predict(subvectors)

        return cluster_ids, pq_codes

    def add(self, X):
        # Encode the vectors and add to the inverted lists
        cluster_ids, pq_codes = self.encode(X)
        for idx, cluster_id in enumerate(cluster_ids):
            self.inverted_lists[cluster_id].append(pq_codes[idx])

    def search(self, queries, nprobe=3, top_k=5):
        # Step 1: Find the nearest IVF clusters for each query
        cluster_distances = self.ivf.transform(queries)
        nearest_clusters = np.argsort(cluster_distances, axis=1)[:, :nprobe]

        # Step 2: Compare within selected clusters using PQ codes
        results = []
        for query_idx, query in enumerate(queries):
            candidates = []
            for cluster in nearest_clusters[query_idx]:
                if cluster in self.inverted_lists:
                    for pq_code in self.inverted_lists[cluster]:
                        dist = 0
                        for i in range(self.m):
                            centroid = self.pq_codebooks[i].cluster_centers_[pq_code[i]]
                            subvector = query[i * self.subvector_dim: (i + 1) * self.subvector_dim]
                            dist += np.linalg.norm(subvector - centroid) ** 2
                        candidates.append(dist)

            # Step 3: Sort the candidates by distance and return the top_k results
            candidates.sort()
            results.append(candidates[:top_k])

        return results


# Example usage
# if __name__ == "__main__":
#     d = 128  # Dimensionality of the vectors
#     nb = 1000  # Number of database vectors
#     nq = 5  # Number of query vectors

#     # Generate some random data
#     np.random.seed(123)
#     X = np.random.random((nb, d)).astype('float32')  # Database vectors
#     queries = np.random.random((nq, d)).astype('float32')  # Query vectors

#     # Initialize and train the IVFPQ index
#     ivfpq = IVFPQ(d=d, nlist=10, m=8, bits_per_subvector=4)
#     ivfpq.train(X)
#     ivfpq.add(X)

#     # Search for the top_k nearest neighbors for the query vectors
#     results = ivfpq.search(queries)

#     # Display the search results
#     for i, res in enumerate(results):
#         print(f"Query {i}:")
#         for dist, pq_code in res:
#             print(f"  Distance: {dist}")
