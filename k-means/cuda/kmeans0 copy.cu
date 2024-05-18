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

__device__ float euclideanDistance(float *point, float *centroid, int nDimensions){
    float sum = 0;
    for(int i = 0; i < nDimensions; i++){
        sum += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sqrt(sum);
}
__global__ void closestCentroid(float *points, float *centroids, int *labels, int nPoints, int nDimensions, int nCentroids){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nPoints){
        float minDistance = euclideanDistance(points + index * nDimensions, centroids, nDimensions);
        labels[index] = 0;
        for(int i = 1; i < nCentroids; i++){
            float distance = euclideanDistance(points + index * nDimensions, centroids + i * nDimensions, nDimensions);
            if(distance < minDistance){
                minDistance = distance;
                labels[index] = i;
            }
        }
    }
}

__global__ void updateCentroids(float *points, float *centroids, float *oldCentroids, int *counts, int *labels, int nPoints, int nDimensions, int nCentroids, float* error_val){
    extern __shared__ float error[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nPoints){
        int label = labels[index];
        for(int i = 0; i < nDimensions; i++){
            atomicAdd(centroids + label * nDimensions + i, points[index * nDimensions + i]);
        }
        atomicAdd(counts + label, 1);
    }
    // calculate mean and error
    if(index < nCentroids * nDimensions){
        int label = index / nDimensions;
        centroids[index] /= counts[label];
        error[index] = abs(centroids[index] - oldCentroids[index]);
    }
    // apply reduction get error in error[0]
    if(index < nCentroids * nDimensions){
        int n = nCentroids * nDimensions;
        for (int stride = n / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (index < stride) {
                error[index] += error[index + stride];
            }
        }
    }
    if(index == 0){
        *error_val = error[0];
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


void kmeans(float * points, float * &centroids, int * &labels,  int nPoints, int nDimensions, int nCentroids, int maxIters){
    // Device Data
    float *d_points, *d_centroids, *d_oldCentroids, *error_val;
    int *d_labels, *d_counts;

    // Allocate memory on GPU
    cudaMalloc(&d_points, nPoints * nDimensions * sizeof(float));
    cudaMalloc(&d_centroids, nCentroids * nDimensions * sizeof(float));
    cudaMalloc(&d_oldCentroids, nCentroids * nDimensions * sizeof(float));
    cudaMalloc(&error_val, sizeof(float));
    cudaMalloc(&d_labels, nPoints * sizeof(int));
    cudaMalloc(&d_counts, nCentroids * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_points, points, nPoints * nDimensions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldCentroids, centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyHostToDevice);



}

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
    // printData(nPoints, nDimensions, nCentroids, maxIters, points, centroids);
    int *labels = (int*)malloc(nPoints * sizeof(int));
    kmeans(points, centroids, labels, nPoints, nDimensions, nCentroids, maxIters);

    
}