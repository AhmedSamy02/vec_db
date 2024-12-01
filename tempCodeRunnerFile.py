data = np.random.random((1000, 128))

# Initialize IVFPQ with 10 clusters, 4 subvectors, and 8 bits per subvector
ivfpq = CustomIVFPQ(d=128,nlist=10, m=8, bits_per_subvector=4)

# Fit the model (cluster the data and apply PQ)
ivfpq.train(data)
# ivfpq.encode(data)

# Encode the entire dataset
encoded_data = ivfpq.encode(data)

# Generate a random query vector (of the same dimension as the data)
query = np.random.random(128)

# Search for the 2 most similar vectors to the query
results = ivfpq.search(query, top_k=2)
print(results)