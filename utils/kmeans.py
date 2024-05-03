import numpy as np
import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]
# print(inputFile, outputFile)
n_points = 0
n_features = 0
n_clusters = 0
max_iter = 0
points = None
centroids = None
with open(inputFile, 'r') as f:
    n_points, n_features, n_clusters, max_iter = map(int, f.readline().split())
    centroids = np.array([list(map(float, f.readline().split())) for _ in range(n_clusters)]).astype(np.float32)
    points = np.array([list(map(float, f.readline().split())) for _ in range(n_points)]).astype(np.float32)
    # print(centroids)


def kmeans(n_points, n_features, n_clusters, max_iter, output_file, result_file):
    old_centroids = centroids.copy()
    for i in range(max_iter):
        print(f"iteration {i}")
        labels = np.argmin(np.linalg.norm(points[:, None] - centroids, axis=2), axis=1)
        # print(labels)
        # update centroids
        for j in range(n_clusters):
            centroids[j] = np.mean(points[labels == j], axis=0)
        # error is abs sum of difference between old and new centroids
        error = np.abs(np.sum(centroids - old_centroids))
        print(f"Error: {error}")
        if error < 1e-6:
            print(f"Converged after {i} iterations")
            break

    with open(result_file, 'w') as f:
        for i in range(n_clusters):
            f.write(" ".join([str(x) for x in centroids[i]]) + "\n")
        for i in range(n_points):
            f.write(f"{labels[i]}\n")

kmeans(n_points=n_points, n_features=n_features, n_clusters=n_clusters, max_iter=max_iter, output_file=outputFile, result_file=outputFile)