# import numpy as np
# from sklearn.cluster import KMeans

# class OPQ:
#     def __init__(self, M, Ks):
#         self.M = M  # Number of subspaces
#         self.Ks = Ks  # Number of clusters per subspace
#         self.R = None  # Rotation matrix
#         self.codebooks = None  # Codebooks for each subspace

#     def fit(self, X):
#         n, d = X.shape
#         assert d % self.M == 0, "The dimension of the input data must be divisible by M"
#         subspace_dim = d // self.M

#         # Initialize rotation matrix as identity
#         self.R = np.eye(d)

#         # Rotate data
#         X_rotated = X @ self.R

#         # Initialize codebooks
#         self.codebooks = []

#         for m in range(self.M):
#             subspace_data = X_rotated[:, m*subspace_dim:(m+1)*subspace_dim]
#             kmeans = KMeans(n_clusters=self.Ks).fit(subspace_data)
#             self.codebooks.append(kmeans.cluster_centers_)

#     def encode(self, X):
#         n, d = X.shape
#         subspace_dim = d // self.M
#         X_rotated = X @ self.R
#         codes = np.zeros((n, self.M), dtype=np.int)

#         for m in range(self.M):
#             subspace_data = X_rotated[:, m*subspace_dim:(m+1)*subspace_dim]
#             distances = np.linalg.norm(subspace_data[:, np.newaxis] - self.codebooks[m], axis=2)
#             codes[:, m] = np.argmin(distances, axis=1)

#         return codes

#     def decode(self, codes):
#         n, _ = codes.shape
#         subspace_dim = self.codebooks[0].shape[1]
#         X_rotated = np.zeros((n, self.M * subspace_dim))

#         for m in range(self.M):
#             X_rotated[:, m*subspace_dim:(m+1)*subspace_dim] = self.codebooks[m][codes[:, m]]

#         X = X_rotated @ self.R.T
#         return X