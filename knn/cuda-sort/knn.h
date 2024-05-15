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

extern int k, n, dim;

// IO functions
bool read_data(string input_file, float*& data, int*& labels, float*& target);
bool write_data(string output_file, float* output, int* labelsOutput);
void print_top(float* data, int* labels, int n, float* target);

// CUDA functions
__global__ void mergeSort(float* data, int* labels, float* target, long long n, int dim, long long threadSize, long long sortedSize);
__global__ void bubbleSort(float* data, int* labels, int n, int dim, float* target, int sizeToSort);
__global__ void printArr(float* data, int n, int dim , float* target);

#endif