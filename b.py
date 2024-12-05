import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from numpy.linalg import norm
import pickle
class MergedIVFPQ:
    def __init__(self, d, nlist, m, bits_per_subvector):
        """
        Initialize the IVFPQ index.
        :param d: Dimensionality of the input vectors
        :param nlist: Number of Voronoi cells (clusters) in IVF
        :param m: Number of sub-vectors for Product Quantization (PQ)
        :param bits_per_subvector: Number of bits for each subvector's quantization
        """
        self.d = d
        self.nlist = nlist
        self.m = m
        self.k = 2 ** bits_per_subvector  # Number of centroids per subvector
        self.subvector_dim = d // m

        assert d % m == 0, "Dimensionality must be divisible by the number of subvectors (m)."

        # Initialize KMeans for IVF and PQ
        self.kmeans = KMeans(n_clusters=self.nlist, init='k-means++', n_init='auto' )
        self.pq_codebooks = [KMeans(n_clusters=self.k,init='k-means++', n_init='auto') for _ in range(m)]

        self.inverted_lists = {i: [] for i in range(nlist)}  # Inverted file structure

    def train(self, data):
        """
        Train the IVF and PQ models on the given data.
        :param data: Input data array of shape (N, D)
        """
        self.kmeans.fit(data)
        self.centroids = self.kmeans.cluster_centers_
        
        # Form the codebook for each subvector
        for i in range(self.m):
            subvectors = data[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
            self.pq_codebooks[i].fit(subvectors)

    def encode(self, data):
        """
        Encode data into PQ codes.
        :param data: Input data array of shape (N, D)
        :return: PQ codes for the data
        """
        cluster_ids = self.kmeans.predict(data)  # Predict cluster IDs for all data
        pq_codes = np.zeros((data.shape[0], self.m), dtype=np.int32) 

        for idx, cluster_id in enumerate(cluster_ids):
            residual = data[idx] - self.centroids[cluster_id]
            for i in range(self.m):
                subvector = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
                centroid_vectors = self.pq_codebooks[i].cluster_centers_

                subvector_norm = norm(subvector)
                centroid_norms = norm(centroid_vectors, axis=1)

                if subvector_norm > 0:
                    similarities = np.dot(centroid_vectors, subvector.T) / (centroid_norms[:, np.newaxis] * subvector_norm)
                    similarities[np.isnan(similarities)] = -np.inf  # Handle NaN values
                    pq_codes[idx, i] = np.argmax(similarities)
                else:
                    pq_codes[idx, i] = 0  # Assign a default centroid if the subvector is zero
# pq_code stores the index for the codebook centers 
            self.inverted_lists[cluster_id].append((idx, pq_codes[idx]))

        return pq_codes

    def encode_single(self, vector):
        """
        Encode a single query vector.
        :param vector: Query vector of shape (D,)
        :return: (cluster_id, pq_code) for the query vector
        """
        cluster_id = self.kmeans.predict(vector.reshape(1, -1))[0]
        residual = vector - self.centroids[cluster_id]
        pq_code = np.zeros((self.m,), dtype=np.int32)

        for i in range(self.m):
            subvector = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
            centroid_vectors = self.pq_codebooks[i].cluster_centers_

            subvector_norm = norm(subvector)
            centroid_norms = norm(centroid_vectors, axis=1)

            if subvector_norm > 0:
                similarities = np.dot(centroid_vectors, subvector.T) / (centroid_norms[:, np.newaxis] * subvector_norm)
                similarities[np.isnan(similarities)] = -np.inf  # Handle NaN values
                pq_code[i] = np.argmax(similarities)
            else:
                pq_code[i] = 0  # Assign a default centroid if the subvector is zero

        return cluster_id, pq_code

    def search(self, query, top_k, nprobe=1):
        """
        Search for the top_k nearest neighbors to the query.
        :param query: Query vector of shape (1, D)
        :param top_k: Number of nearest neighbors to return
        :param nprobe: Number of clusters to probe
        :return: List of indices of the nearest neighbors
        """
        query = query.reshape(1, -1)
        
        # Calculate distances to each centroid and get the closest clusters
        cluster_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(cluster_distances)[:nprobe]
        nearest_centroid = self.centroids[nearest_clusters[0]]
        residual = query - nearest_centroid  # Residual vector
        pq_similarities = []

        # Iterate over the nearest clusters
        for cluster_id in nearest_clusters:
            candidates = self.inverted_lists[cluster_id]
            
            for idx, pq_code in candidates:
                # Reconstruct the vector from the PQ code
                reconstructed_vector = self.reconstruct(pq_code, cluster_id) # center values for the vector
                
                # Calculate similarity (or distance) between query and reconstructed vector
                similarity = self.compute_similarity(residual, reconstructed_vector)
                
                # Append the result (index and similarity)
                pq_similarities.append((idx, similarity))
        
        # Sort by similarity in descending order
        pq_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the indices of the top_k nearest neighbors
        return [idx for idx, _ in pq_similarities[:top_k]]

    def reconstruct(self, pq_code, cluster_id):
        """
        Reconstruct the original vector from the PQ code.
        :param pq_code: The PQ code (list of centroid indices for each subvector)
        :param cluster_id: The cluster ID where the data point belongs (used to access centroids)
        :return: The reconstructed vector
        """
        reconstructed_vector = []
        
        # Reconstruct the vector by taking centroids corresponding to the PQ code
        for i in range(self.m):
            centroid_sub = self.pq_codebooks[i].cluster_centers_[pq_code[i]]
            reconstructed_vector.append(centroid_sub)
        
        return np.hstack(reconstructed_vector)  # Concatenate subvectors to form the full vector

    def compute_similarity(self, query, reconstructed_vector):
        """
        Compute the similarity (e.g., cosine similarity) between the query and the reconstructed vector.
        :param query: The query vector
        :param reconstructed_vector: The reconstructed vector from the PQ code
        :return: The computed similarity
        """
        dot_product = np.dot(query, reconstructed_vector)
        norm_query = norm(query)
        norm_reconstructed = norm(reconstructed_vector)
        
        if norm_query > 0 and norm_reconstructed > 0:
            similarity = dot_product / (norm_query * norm_reconstructed)
        else:
            similarity = -1  # If either vector has zero norm, assign a very low similarity
        
        return similarity

    def save_model(self, filepath='ivfpq_model.dat'):
        """
        Save the trained IVFPQ model to a .dat file.
        :param filepath: Path to the file where the model will be saved
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'centroids': self.centroids,
                'pq_codebooks': [codebook.cluster_centers_ for codebook in self.pq_codebooks],
                'inverted_lists': self.inverted_lists
            }, f)

    def load_model(self, filepath='ivfpq_model.dat'):
        """
        Load the IVFPQ model from a .dat file.
        :param filepath: Path to the file where the model is saved
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.centroids = data['centroids']

            # Reinitialize and set the PQ codebook centroids
            for i, codebook_centers in enumerate(data['pq_codebooks']):
                self.pq_codebooks[i].cluster_centers_ = codebook_centers

            self.inverted_lists = data['inverted_lists']

# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from numpy.linalg import norm

# class MergedIVFPQ:
#     def __init__(self, d, nlist, m, bits_per_subvector):
#         self.d = d
#         self.nlist = nlist
#         self.m = m
#         self.k = 2 ** bits_per_subvector
#         self.subvector_dim = d // m

#         assert d % m == 0, "Dimensionality must be divisible by the number of subvectors (m)."

#         # MiniBatchKMeans for incremental clustering
#         self.kmeans = MiniBatchKMeans(n_clusters=self.nlist, batch_size=256, max_iter=10, n_init='auto')
#         self.pq_codebooks = [
#             MiniBatchKMeans(n_clusters=self.k, batch_size=256, max_iter=10, n_init='auto') for _ in range(m)
#         ]
#         self.inverted_lists = {i: [] for i in range(self.nlist)}

#     def train_batch(self, batch_vectors):
#         """
#         Train the IVF and PQ models using an array of batch vectors.
#         :param batch_vectors: An array of shape (batch_size, d)
#         """
#         # Train the k-means for IVF in batches
#         self.kmeans.partial_fit(batch_vectors)
#         self.centroids = self.kmeans.cluster_centers_

#         # Train each PQ codebook on corresponding subvectors
#         for i in range(self.m):
#             subvectors = batch_vectors[:, i * self.subvector_dim: (i + 1) * self.subvector_dim]
#             self.pq_codebooks[i].partial_fit(subvectors)

#     def encode_batch(self, batch_vectors):
#         """
#         Encode vectors in batches and populate the inverted lists.
#         :param batch_vectors: An array of shape (batch_size, d)
#         """
#         cluster_ids = self.kmeans.predict(batch_vectors)  # Predict cluster index for each vector
#         pq_codes = np.zeros((batch_vectors.shape[0], self.m), dtype=np.int32)

#         for idx, cluster_id in enumerate(cluster_ids):
#             residual = batch_vectors[idx] - self.centroids[cluster_id]
#             for i in range(self.m):
#                 subvector = residual[i * self.subvector_dim: (i + 1) * self.subvector_dim].reshape(1, -1)
#                 centroid_vectors = self.pq_codebooks[i].cluster_centers_

#                 subvector_norm = norm(subvector)
#                 centroid_norms = norm(centroid_vectors, axis=1)

#                 if subvector_norm > 0:
#                     similarities = np.dot(centroid_vectors, subvector.T) / (
#                         centroid_norms[:, np.newaxis] * subvector_norm
#                     )
#                     similarities[np.isnan(similarities)] = -np.inf
#                     pq_codes[idx, i] = np.argmax(similarities)
#                 else:
#                     pq_codes[idx, i] = 0  # Default centroid if subvector is zero

#             self.inverted_lists[cluster_id].append((idx, pq_codes[idx]))

#     def search(self, query, top_k, nprobe=1):
#         query = query.reshape(1, -1)
#         cluster_distances = np.linalg.norm(self.centroids - query, axis=1)
#         nearest_clusters = np.argsort(cluster_distances)[:nprobe]

#         pq_similarities = []
#         for cluster_id in nearest_clusters:
#             candidates = self.inverted_lists[cluster_id]
#             for idx, pq_code in candidates:
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
