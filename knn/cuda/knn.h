#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <map>
using namespace std;

extern int k, n, dim;

// IO functions
bool read_data(string input_file, double*& data, int*& labels, double*& target);
bool write_data(string output_file, double* output, int* labelsOutput);
void print_top(double* data, int* labels, int n, double* target);

// CUDA functions
__global__ void knn(double* data, int* labels, int threadSize, int n, int dim,
    int k, double* target, double* output, int* labelsOutput);
#endif