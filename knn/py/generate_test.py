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
