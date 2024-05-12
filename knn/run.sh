#!/bin/bash

# parameters: $1 = 10000, $2 = 2
test_file="./tests/test_${1}_${2}.txt"
out_file="./out/out_${1}_${2}.txt"
prof_file="./prof/prof_${1}_${2}.prof"
exec_file="./knn.out"

# Function to write a string in yellow
print_yellow() {
    local string="$1"
    echo -e "\e[33m$string\e[0m"
}

print_yellow "Compiling knn.cu\n"
nvcc cuda-sort/*.cu -o "$exec_file"
print_yellow "Running $exec_file with $test_file and saving output to $out_file\n"
"./$exec_file" "$test_file" "$out_file"
# nvprof "./$exec_file" "$test_file" "$out_file"
# nvprof -o "$prof_file" "$exec_file" "$test_file" "$out_file"
