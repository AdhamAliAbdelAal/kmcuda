import numpy as np
from sklearn.cluster import kmeans_plusplus
import sys
np.random.seed(0)

n_points = int(sys.argv[1])
n_features = int(sys.argv[2])
n_clusters = int(sys.argv[3])
max_iter = int(sys.argv[4])
output_file = sys.argv[5]
result_file = sys.argv[6]

# centroids = (np.random.rand(n_clusters, n_features).astype(np.float32)*1000).astype(np.int32)
limit = -1000
points = np.random.uniform(-limit,limit,(n_points, n_features)).astype(np.float32)
centroids,_ = kmeans_plusplus(points, n_clusters)

# write those to output file
with open(output_file, 'w') as f:
    f.write(f"{n_points} {n_features} {n_clusters} {max_iter}\n")
    for i in range(n_clusters):
        f.write(" ".join([str(x) for x in centroids[i]]) + "\n")
    for i in range(n_points):
        f.write(" ".join([str(x) for x in points[i]]) + "\n")
