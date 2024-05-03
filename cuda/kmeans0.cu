#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstdio>

#define N 10000000
#define MAX_ERR 1e-6
__global__ void MatAdd(float* A, float* B, float* C,int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n)
    {
        C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
}

float * allocateMatrix(int n, int m) {
    float * matrix = (float *)malloc(n * m * sizeof(float));
    return matrix;
}

void freeMatrix(float * matrix, int n) {
    free(matrix);
}

void readMatrix(FILE* file, float* A, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            fscanf(file, "%f", &A[i*n+j]);
        }
    }
}

float* allocateAndCopyMatrixToDevice(float* mat, int m, int n){
    float* d_mat;
    cudaMalloc(&d_mat, m * n * sizeof(float));
    cudaMemcpy(d_mat, mat, m * n * sizeof(float), cudaMemcpyHostToDevice);
    return d_mat;
}

void deallocateDeviceMatrix(float* d_mat, int m){
    cudaFree(d_mat);
}

void copyFromDevice(float* mat, float* d_mat, int m, int n){
    cudaMemcpy(mat, d_mat, m * n * sizeof(float), cudaMemcpyDeviceToHost);
}
void outputMatrix(char name ,float* A, int m, int n, FILE* output){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            fprintf(output, "%f ", A[i*n+j]);
        }
        fprintf(output, "\n");
    }
}
// void solve(FILE* file, FILE* output){
//     float* A;
//     float* B;
//     int m, n;
//     readData(file, A, B, m, n);
//     // printf("m = %d, n = %d\n", m, n);
//     // outputMatrix('A', A, m, n);
//     // outputMatrix('B', B, m, n);
//     float* C = allocateMatrix(m, n);
//     float* d_A = allocateAndCopyMatrixToDevice(A, m, n);
//     float* d_B = allocateAndCopyMatrixToDevice(B, m, n);
//     float* d_C = allocateAndCopyMatrixToDevice(C, m, n);
//     dim3 threadsPerBlock(16, 16);
//     dim3 numBlocks(ceil(m/16.0), ceil(n/16.0));
//     MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n);
//     copyFromDevice(C, d_C, m, n);
//     outputMatrix('C', C, m, n, output);
//     deallocateDeviceMatrix(d_A, m);
//     deallocateDeviceMatrix(d_B, m);
//     deallocateDeviceMatrix(d_C, m);
//     freeMatrix(A, m);
//     freeMatrix(B, m);
//     freeMatrix(C, m);
// }

FILE* openFile(char* filename, char* mode){
    FILE* file = fopen(filename, mode);
    if (file == NULL){
        printf("Error: file not found\n");
        exit(1);
    }
    return file;
}

void readData(FILE* file, int& nPoints, int& nDimensions, int& nCentroids, int& maxIters, float*& points, float*& centroids){
    fscanf(file, "%d %d %d %d", &nPoints, &nDimensions, &nCentroids, &maxIters);
    centroids = allocateMatrix(nCentroids, nDimensions);
    points = allocateMatrix(nPoints, nDimensions);
    readMatrix(file, centroids, nCentroids, nDimensions);
    readMatrix(file, points, nPoints, nDimensions);
}

void printData(int nPoints, int nDimensions, int nCentroids, int maxIters, float* points, float* centroids){
    printf("nPoints = %d, nDimensions = %d, nCentroids = %d, maxIters = %d\n", nPoints, nDimensions, nCentroids, maxIters);
    printf("Centroids:\n");
    for(int i = 0; i < nCentroids; i++){
        for(int j = 0; j < nDimensions; j++){
            printf("%f ", centroids[i*nDimensions+j]);
        }
        printf("\n");
    }
    printf("Points:\n");
    for(int i = 0; i < nPoints; i++){
        for(int j = 0; j < nDimensions; j++){
            printf("%f ", points[i*nDimensions+j]);
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]){
    if (argc != 3)
    {
        printf("Usage: %s <input file path> <output file path> \n", argv[0]);
        return 1;
    }
    char* inputFileName = argv[1];
    char* outputFilename = argv[2];
    FILE* inputFile = openFile(inputFileName, "r");
    FILE* outputFile = openFile(outputFilename, "w");
    int nPoints, nDimensions, nCentroids, maxIters;
    float* points, *centroids;
    readData(inputFile, nPoints, nDimensions, nCentroids, maxIters, points, centroids);
    printData(nPoints, nDimensions, nCentroids, maxIters, points, centroids);
    
}