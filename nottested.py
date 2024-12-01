from sklearn.cluster import MiniBatchKMeans  # Use for chunk-based training
import numpy as np
from numpy.linalg import norm

class CustomIVFPQ:
    def __init__(self, d, nlist, m, bits_per_subvector):
        self.d = d
        self.nlist = nlist
        self.m = m
        self.k = 2 ** bits_per_subvector
        self.subvector_dim = d // m

        assert d % m == 0, "Dimensionality must be divisible by m."

        # Use MiniBatchKMeans for chunk-based clustering
        self.kmeans = MiniBatchKMeans(n_clusters=self.nlist, batch_size=1000, random_state=42)
        self.centroids = []
        
        # PQ codebooks for each subvector
        self.pq_codebooks = [MiniBatchKMeans(n_clusters=self.k, batch_size=1000, random_state=42) for _ in range(m)]
        
        self.inverted_lists = {i: [] for i in range(nlist)}
        self.PQinverted_lists = {i: [] for i in range(nlist)}

    def train(self, data_chunk):
        """Train using data in chunks."""
        data_chunk = data_chunk.astype(np.float64)
        self.kmeans.partial_fit(data_chunk)  # Incrementally fit KMeans
        self.centroids = self.kmeans.cluster_centers_

        labels = self.kmeans.predict(data_chunk)
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(data_chunk[idx])
        
        # Train PQ codebooks incrementally on each subvector partition
        for cluster_id, vectors in self.inverted_lists.items():
            if len(vectors) == 0:
                continue
            cluster_data = np.array(vectors)
            for i in range(self.m):
                subvectors = cluster_data[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
                self.pq_codebooks[i].partial_fit(subvectors)

    def encode(self, data_chunk):
        """Encode data in chunks."""
        data_chunk = data_chunk.astype(np.float64)
        cluster_ids = self.kmeans.predict(data_chunk)

        pq_codes = np.zeros((data_chunk.shape[0], self.m), dtype=np.int32)
        for idx, cluster_id in enumerate(cluster_ids):
            centroid = self.centroids[cluster_id]
            residual = data_chunk[idx] - centroid

            for i in range(self.m):
                subvectors = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
                pq_codes[idx, i] = self.pq_codebooks[i].predict(subvectors)

            self.PQinverted_lists[cluster_id].append((idx, pq_codes[idx]))

        return cluster_ids, pq_codes

    def encode_Single(self, vector):
        """Encode a single vector."""
        vector = vector.astype(np.float64).reshape(1, -1)
        cluster_id = self.kmeans.predict(vector)[0]
        centroid = self.centroids[cluster_id]
        residual = vector - centroid

        pq_codes = np.zeros((1, self.m), dtype=np.int32)
        for i in range(self.m):
            subvectors = residual[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
            pq_codes[0, i] = self.pq_codebooks[i].predict(subvectors)

        return pq_codes

    def search(self, vector, top_k, nprobe=1):
        """Search for the top_k closest vectors to the query."""
        vector = vector.astype(np.float64).reshape(1, -1)
        distances = self.kmeans.transform(vector)
        cluster_ids = np.argsort(distances, axis=1)[:, :nprobe].flatten()

        vectorCode = self.encode_Single(vector)
        similarities = []

        for cluster_id in cluster_ids:
            data_codes = self.PQinverted_lists[cluster_id]
            for index, code in data_codes:
                norm_vectorCode = norm(vectorCode)
                norm_code = norm(code)

                similarity = 0 if (norm_vectorCode == 0 or norm_code == 0) else np.dot(vectorCode, code) / (norm_vectorCode * norm_code)
                similarities.append((index, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [index for index, _ in similarities[:top_k]]
        return top_k_indices
# Initialize IVFPQ
ivfpq = CustomIVFPQ(d=128, nlist=10, m=8, bits_per_subvector=4)

# Simulate training in chunks
for chunk in [np.random.random((500, 128)) for _ in range(4)]:
    ivfpq.train(chunk)

# Encode a new data chunk
encoded_data = ivfpq.encode(np.random.random((100, 128)))

# Search with a query vector
query = np.random.random(128)
results = ivfpq.search(query, top_k=2)
print(results)
