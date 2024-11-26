# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from numpy.linalg import norm
# class IVFPQ:
#     def __init__(self, nlist, nsubvectors, nbits):
#         self.nlist = nlist  # Number of clusters in IVF
#         self.nsubvectors = nsubvectors  # Number of subvectors in PQ
#         self.nbits = nbits  # Number of bits per subvector in PQ
#         self.centroids = None  # Cluster centroids
#         self.subquantizers = []  # KMeans models for each subvector
#         self.inverted_lists = []  # Inverted lists for each centroid

#     def fit(self, data):
#         # Step 1: Clustering data into nlist clusters (IVF step)
#         kmeans = KMeans(n_clusters=self.nlist)
#         kmeans.fit(data)
#         self.centroids = kmeans.cluster_centers_

#         # Step 2: Creating inverted lists for each cluster
#         self.inverted_lists = [[] for _ in range(self.nlist)]
#         labels = kmeans.labels_
#         for idx, label in enumerate(labels):
#             self.inverted_lists[label].append(data[idx])

#         # Step 3: Product Quantization (PQ) - split data into subvectors and quantize each
#         # We'll create subquantizers for each subvector
#         for i in range(self.nsubvectors):
#             subquantizer = KMeans(n_clusters=2**self.nbits)
#             subdata = data[:, i::self.nsubvectors]  # Select every nth column for each subvector
#             subquantizer.fit(subdata)
#             self.subquantizers.append(subquantizer)

#     def encode(self, data):
#         # Encodes the entire dataset into quantized vectors using PQ
#         encoded_data = []
#         for vector in data:
#             encoded_vector = []
#             for i, subquantizer in enumerate(self.subquantizers):
#                 subvector = vector[i::self.nsubvectors]  # Extract subvector for the i-th subquantizer
#                 encoded_vector.append(subquantizer.predict([subvector])[0])  # Encode subvector
#             encoded_data.append(encoded_vector)
#         return encoded_data


#     def encode_single(self, vector):
#         encoded_vector = []

#         # If the vector is 1D, reshape it to 2D (1, n_features)
#         if len(vector.shape) == 1:
#             vector = vector.reshape(1, -1)

#         vector_size = vector.shape[1]  # Get the size of the vector (e.g., 70)

#         # Ensure the vector size is divisible by nsubvectors
#         if vector_size % self.nsubvectors != 0:
#             raise ValueError(f"Vector size {vector_size} is not divisible by nsubvectors {self.nsubvectors}.")

#         subvector_size = vector_size // self.nsubvectors  # Size of each subvector

#         for i, subquantizer in enumerate(self.subquantizers):
#             # Extract the subvector (slice the vector properly)
#             start_idx = i * subvector_size
#             end_idx = (i + 1) * subvector_size  # Ensure not exceeding vector length
#             subvector = vector[0, start_idx:end_idx]  # Extract the subvector

#             # If the subvector is empty, handle it gracefully
#             if len(subvector) == 0:
#                 raise ValueError(f"Subvector {i} has no features. Check slicing logic.")

#             subvector = np.asarray(subvector, dtype=np.float32)  # Explicitly cast to float32
#             subvector = subvector.reshape(1, -1)  # Ensure reshaping

#             # print(f"subvector dtype: {subvector.dtype}")  # Should print: float64
#             # print(f"subvector shape: {subvector.shape}")  # Should print: (1, n_features)

#             # Predict the quantized label for this subvector
#             encoded_vector.append(subquantizer.predict(subvector)[0])

#         return encoded_vector






#     def search(self, query, top_k=10):
#         # Step 1: Find the closest centroid (inverted index lookup)
#         closest_centroid = np.argmin([np.linalg.norm(query - centroid) for centroid in self.centroids])

#         # Step 2: Retrieve the list of candidates for the closest centroid
#         candidates = self.inverted_lists[closest_centroid]

#         # Step 3: Encode the query using PQ
#         encoded_query = self.encode_single(query)

#         # Step 4: Compare the query to the candidates using Cosine Similarity
#         similarities = []
#         for candidate in candidates:
#             encoded_candidate = self.encode_single(candidate)  # Encode candidate

#             # Compute cosine similarity between encoded_query and encoded_candidate
#             similarity = np.dot(encoded_query, encoded_candidate) / (norm(encoded_query) * norm(encoded_candidate))
#             similarities.append(similarity)

#         # Step 5: Return the top_k closest vectors based on cosine similarity
#         top_k_indices = np.argsort(similarities)[::-1][:top_k]  # Sort in descending order for highest similarity
#         return [candidates[i] for i in top_k_indices]

# # # Example usage
# # # Generating random data: 1000 vectors, each of 128 dimensions
# # data = np.random.random((1000, 128))

# # # Initialize IVFPQ with 10 clusters, 4 subvectors, and 8 bits per subvector
# # ivfpq = IVFPQ(nlist=10, nsubvectors=4, nbits=8)

# # # Fit the model (cluster the data and apply PQ)
# # ivfpq.fit(data)

# # # Encode the entire dataset
# # encoded_data = ivfpq.encode(data)

# # # Generate a random query vector (of the same dimension as the data)
# # query = np.random.random(128)

# # # Search for the 2 most similar vectors to the query
# # results = ivfpq.search(query, top_k=2)
# # print(results)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class IVFPQ:
    def __init__(self, nlist: int, m: int, nbits: int):
        """
        Initializes the IVFPQ (Inverted File with Product Quantization) index.
        
        Parameters:
        - nlist: Number of clusters (partitions of the vector space)
        - m: Number of sub-vectors (divisions of each vector for product quantization)
        - nbits: Number of bits used to encode each sub-vector
        """
        self.nlist = nlist   # Number of clusters (inverted lists)
        self.m = m           # Number of sub-vectors for product quantization
        self.nbits = nbits   # Number of bits for each sub-vector quantization
        
        self.centroids = None
        self.pq_codes = None  # To hold the product quantization codes
        self.inverted_lists = None
        self.product_quantizer = None  # Product quantizer for encoding
    
    def fit(self, data: np.ndarray):
        """
        Fit the IVFPQ model to the given data.
        
        Parameters:
        - data: The data to fit the index on, shape (n_samples, n_features)
        """
        # Perform KMeans clustering on the data to create nlist clusters
        kmeans = KMeans(n_clusters=self.nlist)
        labels = kmeans.fit_predict(data)
        self.centroids = kmeans.cluster_centers_

        # Initialize Product Quantizer and fit it on the data
        self.product_quantizer = ProductQuantizer(self.nbits, self.m)
        self.product_quantizer.fit(data)

        # Apply product quantization to the entire dataset
        self.pq_codes = np.array([self.product_quantizer.encode(data[i:i+1]) for i in range(data.shape[0])])
        
        # Build inverted lists for each cluster
        self.inverted_lists = {i: [] for i in range(self.nlist)}
        for i, label in enumerate(labels):
            self.inverted_lists[label].append(i)
    
    def search(self, query: np.ndarray, top_k: int):
        """
        Search for the top_k nearest neighbors of the query vector using cosine similarity.
        
        Parameters:
        - query: The query vector, shape (1, n_features)
        - top_k: The number of nearest neighbors to return
        
        Returns:
        - A list of indices of the top_k nearest neighbors
        """
        # Assign the query to a cluster (using the centroid of the query vector)
        distances = np.linalg.norm(self.centroids - query, axis=1)
        cluster_id = np.argmin(distances)
        
        # Retrieve the candidates in the nearest cluster
        candidates = self.inverted_lists[cluster_id]
        
        # Quantize the query and compute cosine similarity
        query_codes = self.product_quantizer.encode(query)
        
        # Compute the cosine similarity for each candidate
        candidate_similarities = []
        for idx in candidates:
            candidate_code = self.pq_codes[idx]
        
            # Compute cosine similarity between query and candidate codes
            sim = cosine_similarity(query_codes.reshape(1, -1), candidate_code.reshape(1, -1))[0][0]
            candidate_similarities.append((sim, idx))
        
        # Sort the candidates by cosine similarity in descending order
        candidate_similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return the top_k candidates based on cosine similarity
        return [idx for _, idx in candidate_similarities[:top_k]]


class ProductQuantizer:
    def __init__(self, nbits: int, m: int):
        self.nbits = nbits
        self.m = m
        self.kmeans_models = []  # List to hold KMeans models for each sub-vector
    
    def fit(self, data: np.ndarray):
        """
        Fit the quantizer to the data (use k-means clustering).
        
        Parameters:
        - data: The data to fit, shape (n_samples, n_features)
        """
        # Divide data into sub-vectors
        n_samples, n_features = data.shape
        sub_vector_length = n_features // self.m
        
        # Fit a KMeans model for each sub-vector
        for i in range(self.m):
            sub_vectors = data[:, i * sub_vector_length:(i + 1) * sub_vector_length]
            kmeans = KMeans(n_clusters=2**self.nbits)
            kmeans.fit(sub_vectors)
            self.kmeans_models.append(kmeans)
    
    def encode(self, data: np.ndarray):
        """
        Encode the data into PQ codes.
        
        Parameters:
        - data: The sub-vectors to encode, shape (n_samples, n_features)
        
        Returns:
        - Encoded PQ codes (as integer indices), shape (n_samples, m)
        """
        encoded_data = []
        n_samples, n_features = data.shape
        sub_vector_length = n_features // self.m
        
        # Ensure that the sub-vector data is correctly shaped and typed
        for i in range(self.m):
            sub_vector = data[:, i * sub_vector_length:(i + 1) * sub_vector_length]
            
            # Ensure the sub-vector is of type float64 before predicting
            sub_vector = sub_vector.astype(np.float64)  # Cast to float64
            
            # Ensure it's 2D with shape (n_samples, sub_vector_length)
            sub_vector = sub_vector.reshape(-1, sub_vector.shape[1])
            
            # Perform prediction using the corresponding KMeans model
            encoded_vector = self.kmeans_models[i].predict(sub_vector)
            encoded_data.append(encoded_vector)
        
        # Stack the encoded sub-vectors to form the final encoded representation
        # Return a single vector for the entire data point (each entry is the quantization index of each sub-vector)
        return np.stack(encoded_data, axis=1)  # Shape (n_samples, m)
