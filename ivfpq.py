import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class IVFPQ:
    def __init__(self, nlist, nsubvectors, nbits):
        self.nlist = nlist
        self.nsubvectors = nsubvectors
        self.nbits = nbits
        self.centroids = None
        self.subquantizers = []
        self.inverted_lists = []

    def fit(self, data):
        # Step 1: Clustering data into nlist clusters
        kmeans = KMeans(n_clusters=self.nlist)
        kmeans.fit(data)
        self.centroids = kmeans.cluster_centers_

        # Step 2: Creating inverted lists
        self.inverted_lists = [[] for _ in range(self.nlist)]
        labels = kmeans.labels_
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(data[idx])

        # Step 3: Product Quantization
        for i in range(self.nsubvectors):
            subquantizer = KMeans(n_clusters=2**self.nbits)
            subdata = data[:, i::self.nsubvectors]
            subquantizer.fit(subdata)
            self.subquantizers.append(subquantizer)


    def encode(self, data):
        encoded_data = []
        for vector in data:
            encoded_vector = []
            for i, subquantizer in enumerate(self.subquantizers):
                subvector = vector[i::self.nsubvectors]
                encoded_vector.append(subquantizer.predict([subvector])[0])
            encoded_data.append(encoded_vector)
        return encoded_data

    def search(self, query, top_k=10):
        # Step 1: Find the closest centroid
        closest_centroid = np.argmin([np.linalg.norm(query - centroid) for centroid in self.centroids])
        
        # Step 2: Search within the corresponding inverted list
        candidates = self.inverted_lists[closest_centroid]
        distances = [np.linalg.norm(query - candidate) for candidate in candidates]
        
        # Step 3: Return the top_k closest vectors
        top_k_indices = np.argsort(distances)[:top_k]
        return [candidates[i] for i in top_k_indices]

# Example usage
# data = np.random.random((1000, 128))
# ivfpq = IVFPQ(nlist=10, nsubvectors=4, nbits=8)
# ivfpq.fit(data)
# encoded_data = ivfpq.encode(data)
# query = np.random.random(128)
# results = ivfpq.search(query,2)
# print(results)