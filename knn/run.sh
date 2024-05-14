#!/bin/bash

# Parameters: $1 = 10000, $2 = 2
test_file="./tests/test_${1}_${2}.txt"
out_file="./out/out_${1}_${2}.txt"
exec_file="./knn.out"

# Function to write a string in yellow
print_yellow() {
    local string="$1"
    echo -e "\e[33m$string\e[0m"
}

print_yellow "Compiling knn.cu\n"
nvcc cuda/*.cu -o "$exec_file"

if [ $? -eq 0 ]; then
    print_yellow "Compilation successful. Running $exec_file with $test_file and saving output to $out_file\n"
    "./$exec_file"  "$test_file" "$out_file"
    # cuda-gdb "./$exec_file" -ex "run $test_file $out_file"
else
    echo "Compilation failed."
fi
