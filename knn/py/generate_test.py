import numpy as np


def generate_test_case(array_size, dim=2, n_classes=2):
    # Generate 1 test examples
    test_examples = np.random.rand(dim)

    # Generate the array
    array = np.random.rand(array_size, dim)

    # Generate the labels
    labels = np.random.randint(0, n_classes, array_size)

    return test_examples, array, labels


# Example usage
k = 5
dim = 2
array_size = 10
test_example, array, labels = generate_test_case(array_size, dim=dim)

# Write the result to a file
output_dir = "tests"
output_file = f"{output_dir}/test_{array_size}_{dim}.txt"
with open(output_file, "w") as file:
    # write k
    file.write(f"{k}\n")
    # write array_size and dimension
    file.write(f"{array_size} {dim}\n")
    # write test example
    file.write(" ".join(map(str, test_example)) + "\n")
    # write array and labels
    for i in range(array_size):
        file.write(" ".join(map(str, array[i])) + f" {labels[i]}\n")


print("Result written to file:", output_file)


# k nearest neighbors
# Path: knn/py/knn.py
import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn(X, y, query_point, k=5):
    # Step 1: Compute the distances and indices
    distances = []
    for i in range(X.shape[0]):
        distance = euclidean_distance(X[i], query_point)
        distances.append((distance, y[i], i))

    # Step 2: Sort the distances
    distances = sorted(distances)
    
    # Step 3: Pick the first k points
    distances = distances[:k]

    # Step 4: Get the labels and indices
    labels = np.array([label for _, label, _ in distances])
    indices = np.array([index for _, _, index in distances])

    # Step 5: Get the majority vote
    majority_vote = Counter(labels).most_common(1)[0][0]
    
    return majority_vote, indices

    

# output result to a file
output_dir = "results"
output_file = f"{output_dir}/result_{array_size}_{dim}.txt"
with open(output_file, "w") as file:
    # write the result
    majority_vote, indices = knn(array, labels, test_example, k=k)
    file.write(f"{majority_vote}\n")
    file.write(" ".join(map(str, indices)) + "\n")

