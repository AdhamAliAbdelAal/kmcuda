import numpy as np
from sklearn.cluster import KMeans
import sys

n_points = int(sys.argv[1])
n_features = int(sys.argv[2])
n_clusters = int(sys.argv[3])
max_iter = int(sys.argv[4])
output_file = sys.argv[5]
result_file = sys.argv[6]

# centroids = (np.random.rand(n_clusters, n_features).astype(np.float32)*1000).astype(np.int32)

points = (np.random.rand(n_points, n_features).astype(np.float32)*1000).astype(np.int32)

# # run kmeans to get the labels
# kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=centroids).fit(points)

# # write the final centroids and labels to result file
# with open(result_file, 'w') as f:
#     for i in range(n_clusters):
#         f.write(" ".join([str(x) for x in kmeans.cluster_centers_[i]]) + "\n")
#     for i in range(n_points):
#         f.write(f"{kmeans.labels_[i]}\n")

# write those to output file
with open(output_file, 'w') as f:
    f.write(f"{n_points} {n_features} {n_clusters} {max_iter}\n")
    # for i in range(n_clusters):
    #     f.write(" ".join([str(x) for x in centroids[i]]) + "\n")
    for i in range(n_points):
        f.write(" ".join([str(x) for x in points[i]]) + "\n")
