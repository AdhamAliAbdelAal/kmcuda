import numpy as np
import sys

file1=sys.argv[1]
file2=sys.argv[2]
skip = int(sys.argv[3])

def read_file(file):
    labels= np.array([int(x) for x in open(file).readlines()[skip:]])
    return labels

labels1 = read_file(file1)
labels2 = read_file(file2)

print(np.sum(labels1 == labels2) / len(labels1))