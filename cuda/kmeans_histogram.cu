#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <cstdio>
using namespace std;

#define MAX_ERR 1e-6
#define MAX_CLUSTERS 1000

__device__ float euclideanDistance(float *point, float *centroid, int nDimensions){
    float sum = 0;
    for(int i = 0; i < nDimensions; i++){
        sum += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sum;
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

__global__ void aggregateCentroids(float *points, float *centroids, int *counts, int *labels, int nPoints, int nDimensions, int nCentroids){
    extern __shared__ float centroids_privatization[];
    __shared__ int counts_privatization[MAX_CLUSTERS];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // load the values of counts and centroids into counts_privatization and centroids_privatization
    // initialize counts
    if(threadIdx.x < nCentroids){
        counts_privatization[threadIdx.x] = 0;
    }
    // initialize centroids
    if(threadIdx.x < nCentroids*nDimensions){
        centroids_privatization[threadIdx.x] = 0;
    }
    __syncthreads();
    if(index < nPoints){
        int label = labels[index];
        for(int i = 0; i < nDimensions; i++){
            // atomicAdd(centroids + label * nDimensions + i, points[index * nDimensions + i]);
            atomicAdd(centroids_privatization + label * nDimensions + i, points[index * nDimensions + i]);
        }
        // atomicAdd(counts + label, 1);
        atomicAdd(counts_privatization + label, 1);
    }
    __syncthreads();
    // add privatized values to global memory
    if(threadIdx.x < nCentroids){
        for(int i = 0; i < nDimensions; i++){
            atomicAdd(centroids + threadIdx.x * nDimensions + i, centroids_privatization[threadIdx.x * nDimensions + i]);
        }
        atomicAdd(counts + threadIdx.x, counts_privatization[threadIdx.x]);
    }
}

// given the centroids and the counts, update the centroids and calculate the error between the new and old centroids
__global__ void updateCentroids(float *centroids, float *oldCentroids, int *counts, int nDimensions, int nCentroids, float* error_val){
    extern __shared__ float error[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < nCentroids * nDimensions){
        // each thread update one float in the centroids array
        // mean calculations
        int label = index / nDimensions;
        centroids[index] /= counts[label];
        error[index] = abs(centroids[index] - oldCentroids[index]);

        // reduction step
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


void freeMatrix(float * matrix) {
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

    // Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    int threadsPerBlockCentroids = nCentroids * nDimensions;
    int blocksPerGridCentroids = 1;
    for(int i = 0; i < maxIters; i++){
        cudaMemset(d_counts, 0, nCentroids * sizeof(int));
        printf("Iteration %d\n", i);
        closestCentroid<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_centroids, d_labels, nPoints, nDimensions, nCentroids);
        // printf("Closest Centroid Done\n");
        aggregateCentroids<<<blocksPerGrid, threadsPerBlock, nCentroids * nDimensions * sizeof(float)>>>(d_points, d_centroids, d_counts, d_labels, nPoints, nDimensions, nCentroids);
        // printf("Aggregate Centroids Done\n");
        updateCentroids<<<blocksPerGridCentroids, threadsPerBlockCentroids, nCentroids * nDimensions * sizeof(float)>>>(d_centroids, d_oldCentroids, d_counts, nDimensions, nCentroids, error_val);
        // printf("Update Centroids Done\n");
        float error;
        cudaMemcpy(&error, error_val, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Error: %f\n", error);
        if(error < MAX_ERR){
            printf("Converged\n");
            break;
        }
        cudaMemcpy(d_oldCentroids, d_centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyDeviceToDevice);

    }
    cudaMemcpy(centroids, d_centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, d_labels, nPoints * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Done\n");

    // Free memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_oldCentroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(error_val);
}

FILE* openFile(char* filename, string mode){
    FILE* file = fopen(filename, mode.c_str());
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

void writeData(FILE *file,float *centroids, int *labels, int nPoints, int nDimensions, int nCentroids){
    for(int i = 0; i < nCentroids; i++){
        for(int j = 0; j < nDimensions; j++){
            fprintf(file, "%f ", centroids[i*nDimensions+j]);
        }
        fprintf(file, "\n");
    }
    for(int i = 0; i < nPoints; i++){
        fprintf(file, "%d\n", labels[i]);
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
    // write output
    writeData(outputFile, centroids, labels, nPoints, nDimensions, nCentroids);
    // free memory
    freeMatrix(points);
    freeMatrix(centroids);
    free(labels);
    fclose(inputFile);
    fclose(outputFile);
}