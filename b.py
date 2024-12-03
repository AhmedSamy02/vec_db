# import numpy as np
# from sklearn.cluster import KMeans
# from numpy.linalg import norm

# class MergedIVFPQ:
#     def __init__(self, d, nlist, m, bits_per_subvector):
#         """
#         Initialize the IVFPQ index.
#         :param d: Dimensionality of the input vectors
#         :param nlist: Number of Voronoi cells (clusters) in IVF
#         :param m: Number of sub-vectors for Product Quantization (PQ)
#         :param bits_per_subvector: Number of bits for each subvector's quantization
#         """
#         self.d = d
#         self.nlist = nlist
#         self.m = m
#         self.k = 2 ** bits_per_subvector  # Number of centroids per subvector
#         self.subvector_dim = d // m

#         assert d % m == 0, "Dimensionality must be divisible by the number of subvectors (m)."

#         self.kmeans = KMeans(n_clusters=self.nlist, init='k-means++', n_init='auto')
#         self.pq_codebooks = [KMeans(n_clusters=self.k, init='k-means++', n_init='auto') for _ in range(m)]

#         self.inverted_lists = {i: [] for i in range(nlist)}  # Inverted file structure

#     def train(self, data):
#         """
#         Train the IVF and PQ models on the given data.
#         :param data: Input data array of shape (N, D)
#         """
#         self.kmeans.fit(data)
#         self.centroids = self.kmeans.cluster_centers_
#         # form the code book for all data vectors
#         for i in range(self.m):
#             subvectors = data[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
#             self.pq_codebooks[i].fit(subvectors)

#     def encode(self, data):
#         cluster_ids = self.kmeans.predict(data) # predict all the cluster index for all data
#         pq_codes = np.zeros((data.shape[0], self.m), dtype=np.int32) 

#         for idx, cluster_id in enumerate(cluster_ids):
#             residual = data[idx] - self.centroids[cluster_id]
#             for i in range(self.m):
#                 subvector = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
#                 centroid_vectors = self.pq_codebooks[i].cluster_centers_

#                 # Compute norms and handle zero-norm cases
#                 subvector_norm = norm(subvector)
#                 centroid_norms = norm(centroid_vectors, axis=1)

#                 if subvector_norm > 0:
#                     similarities = np.dot(centroid_vectors, subvector.T) / (centroid_norms[:, np.newaxis] * subvector_norm)
#                     similarities[np.isnan(similarities)] = -np.inf  # Handle NaN values by setting them to very low similarity
#                     pq_codes[idx, i] = np.argmax(similarities)
#                 else:
#                     pq_codes[idx, i] = 0  # Assign a default centroid if the subvector is zero

#             self.inverted_lists[cluster_id].append((idx, pq_codes[idx]))

#     def encode_single(self, vector):
#         cluster_id = self.kmeans.predict(vector.reshape(1, -1))[0]
#         residual = vector - self.centroids[cluster_id]
#         pq_code = np.zeros((self.m,), dtype=np.int32)

#         for i in range(self.m):
#             subvector = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
#             centroid_vectors = self.pq_codebooks[i].cluster_centers_

#             subvector_norm = norm(subvector)
#             centroid_norms = norm(centroid_vectors, axis=1)

#             if subvector_norm > 0:
#                 similarities = np.dot(centroid_vectors, subvector.T) / (centroid_norms[:, np.newaxis] * subvector_norm)
#                 similarities[np.isnan(similarities)] = -np.inf  # Handle NaN values
#                 pq_code[i] = np.argmax(similarities)
#             else:
#                 pq_code[i] = 0  # Assign a default centroid if the subvector is zero

#         return cluster_id, pq_code


#     def search(self, query, top_k, nprobe=1):
#         query = query.reshape(1, -1)
#         cluster_distances = np.linalg.norm(self.centroids - query, axis=1)
#         nearest_clusters = np.argsort(cluster_distances)[:nprobe]

#         pq_similarities = []
#         for cluster_id in nearest_clusters:
#             candidates = self.inverted_lists[cluster_id]
#             for candidate in candidates:
#                 idx, pq_code = candidate
#                 similarity = 0
#                 for i in range(self.m):
#                     query_subvector = query[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
#                     centroid_sub = self.pq_codebooks[i].cluster_centers_[pq_code[i]]

#                     dot_product = np.dot(query_subvector, centroid_sub)
#                     norm_query = norm(query_subvector)
#                     norm_centroid = norm(centroid_sub)

#                     if norm_query > 0 and norm_centroid > 0:
#                         similarity += dot_product / (norm_query * norm_centroid)

#                 pq_similarities.append((idx, similarity))

#         pq_similarities.sort(key=lambda x: x[1], reverse=True)
#         return [idx for idx, _ in pq_similarities[:top_k]]
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from numpy.linalg import norm

class MergedIVFPQ:
    def __init__(self, d, nlist, m, bits_per_subvector):
        self.d = d
        self.nlist = nlist
        self.m = m
        self.k = 2 ** bits_per_subvector
        self.subvector_dim = d // m

        assert d % m == 0, "Dimensionality must be divisible by the number of subvectors (m)."

        # MiniBatchKMeans for incremental clustering with more iterations and a higher number of initializations
        self.kmeans = MiniBatchKMeans(n_clusters=self.nlist, batch_size=256, max_iter=50, n_init=10)
        
        # Each PQ codebook using MiniBatchKMeans
        self.pq_codebooks = [
            MiniBatchKMeans(n_clusters=self.k, batch_size=256, max_iter=50, n_init=10) for _ in range(m)
        ]
        self.inverted_lists = {i: [] for i in range(self.nlist)}

    def train_batch(self, batch_vectors):
        """
        Train the IVF and PQ models using an array of batch vectors.
        :param batch_vectors: An array of shape (batch_size, d)
        """
        # Train the k-means for IVF in batches with more iterations and better initialization
        self.kmeans.partial_fit(batch_vectors)
        self.centroids = self.kmeans.cluster_centers_

        # Train each PQ codebook on corresponding subvectors with more iterations
        for i in range(self.m):
            subvectors = batch_vectors[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
            self.pq_codebooks[i].partial_fit(subvectors)

    def encode_batch(self, batch_vectors):
        """
        Encode vectors in batches and populate the inverted lists.
        :param batch_vectors: An array of shape (batch_size, d)
        """
        # Normalize the batch vectors before encoding (important for stability)
        batch_vectors /= np.linalg.norm(batch_vectors, axis=1, keepdims=True)

        # Predict the cluster index for each vector
        cluster_ids = self.kmeans.predict(batch_vectors)
        pq_codes = np.zeros((batch_vectors.shape[0], self.m), dtype=np.int32)

        for idx, cluster_id in enumerate(cluster_ids):
            residual = batch_vectors[idx] - self.centroids[cluster_id]
            for i in range(self.m):
                subvector = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
                centroid_vectors = self.pq_codebooks[i].cluster_centers_

                subvector_norm = norm(subvector)
                centroid_norms = norm(centroid_vectors, axis=1)

                # Compute similarities and select the most similar centroid
                if subvector_norm > 0:
                    similarities = np.dot(centroid_vectors, subvector.T) / (centroid_norms[:, np.newaxis] * subvector_norm)
                    similarities[np.isnan(similarities)] = -np.inf  # Handle NaN values
                    pq_codes[idx, i] = np.argmax(similarities)
                else:
                    pq_codes[idx, i] = 0  # Default centroid if the subvector is zero

            # Append to the inverted list for this cluster
            self.inverted_lists[cluster_id].append((idx, pq_codes[idx]))

    def search(self, query, top_k, nprobe=1):
        query = query.reshape(1, -1)
        cluster_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(cluster_distances)[:nprobe]

        pq_similarities = []
        for cluster_id in nearest_clusters:
            candidates = self.inverted_lists[cluster_id]
            for idx, pq_code in candidates:
                similarity = 0
                for i in range(self.m):
                    query_subvector = query[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
                    centroid_sub = self.pq_codebooks[i].cluster_centers_[pq_code[i]]

                    dot_product = np.dot(query_subvector, centroid_sub)
                    norm_query = norm(query_subvector)
                    norm_centroid = norm(centroid_sub)

                    if norm_query > 0 and norm_centroid > 0:
                        similarity += dot_product / (norm_query * norm_centroid)

                pq_similarities.append((idx, similarity))

        # Sort by similarity in descending order and return top_k results
        pq_similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in pq_similarities[:top_k]]

# Example of how to use the class
# ivfpq = MergedIVFPQ(d=128, nlist=100, m=8, bits_per_subvector=8)

# # Simulating training on batches
# num_records = 100000  # For example, 100,000 vectors
# batch_size = 1000  # Define your batch size
# data = np.random.rand(num_records, 128)  # Generate some random data

# for i in range(0, num_records, batch_size):
#     batch_vectors = data[i:i + batch_size]
#     ivfpq.train_batch(batch_vectors)
#     ivfpq.encode_batch(batch_vectors)
