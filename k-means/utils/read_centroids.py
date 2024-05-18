import numpy as np
file = 'results.txt'
n_clusters = 4
n_features = 2

# read the centroids into numpy array centroids
with open(file, 'r') as f:
    centroids = np.array([list(map(float, f.readline().split())) for _ in range(n_clusters)]).astype(np.float32)

print(centroids)