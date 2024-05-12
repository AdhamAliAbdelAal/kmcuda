#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <map>
using namespace std;

#define ELEMENTS_PER_THREAD 16

extern int k, n, dim;

// IO functions
bool read_data(string input_file, double*& data, int*& labels, double*& target);
bool write_data(string output_file, double* output, int* labelsOutput);
void print_top(double* data, int* labels, int n, double* target);

// CUDA functions
__global__ void mergeSort(double* data, int* labels, double* target, long long n, int dim, long long threadSize, long long sortedSize);
__global__ void bubbleSort(double* data, int* labels, int n, int dim, double* target, int sizeToSort);
__global__ void printArr(double* data, int n, int dim , double* target);

#endif