#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <map>
using namespace std;

extern long long k, n, dim;

// IO functions
bool read_data(string input_file, double *&data, int *&labels, double *&target, int *&indices);
bool write_data(string output_file, double *output, int *labelsOutput);
void print_top(double *data, int *labels, int n, double *target);

// CUDA functions
__global__ void knn(int *indices, double *distances, int threadSize, long long n, int k, double *target, int *output,
                    double *distancesOut);
__global__ void calcDistances(double *data, double *target, double *distances, long long n, int dim);
#endif