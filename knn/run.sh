#!/bin/bash

# parameters: $1 = 10000, $2 = 2
test_file="./tests/test_${1}_${2}.txt"
out_file="./out/out_${1}_${2}.txt"
prof_file="./prof/prof_${1}_${2}.prof"
exec_file="./knn.out"

nvcc cuda/*.cu -o "$exec_file"
nvprof "./$exec_file" "$test_file" "$out_file"
nvprof -o "$prof_file" "$exec_file" "$test_file" "$out_file"
