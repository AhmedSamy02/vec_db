import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import norm
class CustomIVFPQ:
    def __init__(self, d, nlist, m, bits_per_subvector):
        self.d = d  # Dimensionality
        self.nlist = nlist  # Number of clusters for IVF
        self.m = m  # Number of subvectors for PQ
        self.k = 2 ** bits_per_subvector  # Number of centroids per subvector (codebook size)
        self.subvector_dim = d // m  # Dimensionality per subvector

        assert d % m == 0, "Dimensionality must be divisible by m."
        
        # Remove dtype argument here
        self.kmeans = KMeans(n_clusters=self.nlist,init='k-means++', n_init='auto')
        self.centroids = []
        # KMeans for PQ codebooks, one for each subvector
        self.pq_codebooks = []

        # Inverted lists to store encoded vectors
        self.inverted_lists = {i: [] for i in range(nlist)}
        self.PQinverted_lists = {i: [] for i in range(nlist)}

    def train(self, data):
        data = data.astype(np.float64)  # Ensure training data is float64
        self.kmeans.fit(data)
        self.centroids = self.kmeans.cluster_centers_

        labels = self.kmeans.labels_

        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(data[idx])
        for vectors in self.inverted_lists:
            samples = len(self.inverted_lists[vectors])
            cluster = int(np.floor(np.log2(samples)))
            cluster = min(cluster,self.k)
            kmean = KMeans(n_clusters=cluster,init='k-means++', n_init='auto')
            self.pq_codebooks.append(kmean)
        for cluster_id, vectors in self.inverted_lists.items():
            if len(vectors) == 0:
                continue
            cluster_data = np.array(vectors).astype(np.float64)  # Ensure cluster data is float64

            for i in range(self.m):
                subvectors = cluster_data[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
                self.pq_codebooks[i].fit(subvectors)

    def encode(self, data):
        data = data.astype(np.float64)  # Ensure encoded data is float64
        cluster_ids = self.kmeans.predict(data)
    
        pq_codes = np.zeros((data.shape[0], self.m), dtype=np.int32)
        for idx, cluster_id in enumerate(cluster_ids):
            centroid = self.centroids[cluster_id]
            residual = data[idx] - centroid  # Compute the residual
    
            for i in range(self.m):
                subvectors = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
                pq_codes[idx, i] = self.pq_codebooks[i].predict(subvectors)
    
            self.PQinverted_lists[cluster_id].append((idx, pq_codes[idx]))
    
        return cluster_ids, pq_codes
    
    def encode_Single(self, vector, nprobe=1):
        vector = vector.astype(np.float64).reshape(1, -1)  # Ensure single vector is float64
        cluster_id = self.kmeans.predict(vector)[0]
        centroid = self.centroids[cluster_id]
        residual = vector - centroid  # Compute the residual
    
        pq_codes = np.zeros((1, self.m), dtype=np.int32)
        for i in range(self.m):
            subvectors = residual[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
            pq_codes[0, i] = self.pq_codebooks[i].predict(subvectors)
    
        return pq_codes


    def search(self, vector, top_k, nprobe=1):
        vector = vector.astype(np.float64).reshape(1, -1)  # Ensure vector is float64
        distances = self.kmeans.transform(vector)
        cluster_ids = np.argsort(distances, axis=1)[:, :nprobe].flatten()

        vectorCode = self.encode_Single(vector)
        similarities = []

        for cluster_id in cluster_ids:
            data_codes = self.PQinverted_lists[cluster_id]
            for index, code in data_codes:
                norm_vectorCode = norm(vectorCode)
                norm_code = norm(code)

                if norm_vectorCode == 0 or norm_code == 0:
                    similarity = 0  # Assign 0 similarity if either vector is zero
                else:
                    similarity = np.dot(vectorCode, code) / (norm_vectorCode * norm_code)
                similarities.append((index, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        top_k_indices = [index for index, _ in similarities[:top_k]]
        return top_k_indices


# Test the CustomIVFPQ class
# data = np.random.random((1000, 128))

# # Initialize IVFPQ with 10 clusters, 4 subvectors, and 8 bits per subvector
# ivfpq = CustomIVFPQ(d=128,nlist=10, m=8, bits_per_subvector=4)

# # Fit the model (cluster the data and apply PQ)
# ivfpq.train(data)
# # ivfpq.encode(data)

# # Encode the entire dataset
# encoded_data = ivfpq.encode(data)

# # Generate a random query vector (of the same dimension as the data)
# query = np.random.random(128)

# # Search for the 2 most similar vectors to the query
# results = ivfpq.search(query, top_k=2)
# print(results)