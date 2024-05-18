#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <map>
// import the necessary libraries for file reading
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

#define ELEMENTS_PER_THREAD 16

extern long long n;
extern int k, dim;

// IO functions
bool read_data(string input_file, float *&data, int *&labels, float *&target);
bool write_data(string output_file, float *output, int *labelsOutput);
void print_top(float *data, int *labels, int n, float *target);

// CUDA functions
__global__ void mergeSort(float *data, int *labels, float *distances, long long n, int dim, long long threadSize,
                          long long sortedSize);
__global__ void bubbleSort(float *data, int *labels, float *distances, long long n, int dim, int sizeToSort,
                           long long totalSize);
__global__ void printArr(float *data, int n, int dim, float *target);
__global__ void calcDistances(float *data, float *target, float *distances, long long n, int dim, long long totalSize);

#endif