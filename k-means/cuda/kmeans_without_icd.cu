#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

#define MAX_ERR 1e-6
#define MAX_CLUSTERS 1000
#define LABELING_BLOCK_SIZE 256
#define UPDATE_BLOCK_SIZE 256

__device__ float distance(float *point, float *centroid, int nDimensions){
    float sum = 0;
    for(int i = 0; i < nDimensions; i++){
        sum += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sum;
}
__global__ void labelingKernel(float *points, float *centroids, float* currentCentroids, int *labels, int *counts, int nPoints, int nDimensions, int nCentroids){
    extern __shared__ float centroids_shared[];
    __shared__ int counts_privatization[MAX_CLUSTERS];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // printf("threadIdx: %d\n", threadIdx.x);
    // printf("blockdim: %d\n", blockDim.x);
    // printf("index: %d\n", index);
    // initialize centroids shared memory
    // int loadsPerThread = ceil((float)nCentroids*nDimensions/blockDim.x);
    for(int i=tid;i<nCentroids*nDimensions;i+=blockDim.x){
        centroids_shared[i] = currentCentroids[i];
        // printf("centroids_shared[%d]: %f\n", i, centroids_shared[i]);
    }
    // synchronize threads to ensure copying centroids
    __syncthreads();

    // copy point in register variable to avoid multiple access to global memory
    // float * point = new float[nDimensions]; 
    // float* point = (float*)malloc(nDimensions * sizeof(float));
    // for(int i = 0; i < nDimensions; i++){
    //     point[i] = points[index * nDimensions + i];
    // }  

    float * point = points + index * nDimensions;
    // start labeling
    int label = 0;
    if(index < nPoints){
        float minDistance = distance(point, centroids_shared, nDimensions);
        // use register variable instead of global memory location
        for(int i = 1; i < nCentroids; i++){
            float d = distance(point, centroids_shared + i * nDimensions, nDimensions);
            if(index==1){
                // printf("Distance with centroid %d: %f\n", i, d);
            }
            if(d < minDistance){
                minDistance = d;
                label = i;
            }
        }
        // write the value back to global memory
        labels[index] = label;
        // printf("Point %d: Label %d Distance %f\n", index, label, minDistance);
    }
    // to ensure that all threads calculate the distance before use centroids_shared as the shared memory for privatization
    __syncthreads();

    // initialize privatization arrays in each block
    for (int i = tid; i < nCentroids; i+=blockDim.x){
        counts_privatization[i] = 0;
    }
    // write reset the centroids in the shared memory
    for(int i = tid; i < nCentroids * nDimensions; i+=blockDim.x){
        centroids_shared[i] = 0;
    }
    // sync threads in the block is required to make sure the privatization arrays are initialized
    __syncthreads();

    // add the point to the centroid in the privatization array
    if(index < nPoints){
        // add 1 to the count of the label in the privatization array
        atomicAdd(counts_privatization + label, 1);
        // printf("point %d add to label %d\n", index, label);
        for(int i = 0; i < nDimensions; i++){
            atomicAdd(centroids_shared + label * nDimensions + i, point[i]);
        }
    }

    // sync threads to ensure that all threads have added the point to the privatization arrays
    __syncthreads();
    // write the privatization arrays back to global memory
    for(int i = tid; i < nCentroids; i+=blockDim.x){
        atomicAdd(counts + i, counts_privatization[i]);
    }
    for(int i = tid; i < nCentroids * nDimensions; i+=blockDim.x){
        // printf("centroids_shared[%d]: %f\n", i, centroids_shared[i]);
        atomicAdd(centroids + i, centroids_shared[i]);
    }
    // free the point
    // delete [] point;
}

__global__ void updateKernel(float *centroids, int *counts, float* oldCentroids, float *error, int nDimensions, int nCentroids){
    __shared__ float error_shared[UPDATE_BLOCK_SIZE];
    // printf("hello");
    // [TODO] make the kernel more efficient by using shared memory for counts as it can be used many times in single block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid  = threadIdx.x;
    // initialize the error shared memory
    error_shared[tid] = 0;
    __syncthreads();
    if(index < nCentroids * nDimensions){
        // each thread update one float in the centroids array
        // mean calculations
        float temp = centroids[index];
        // printf("temp before [%d]: %f\n",index, temp);
        int clusterCount = index / nDimensions;
        int cnt = counts[clusterCount];
        // temp = temp?temp:0;
        // cnt = cnt?cnt:1;
        // printf("count[%d]: %d, index %d\n", clusterCount,counts[clusterCount],index);
        temp /= cnt;
        // printf("centroids[%d]: %f\n", index, temp);
        centroids[index] = temp;
        error_shared[tid] = abs(temp - oldCentroids[index]);
        // printf("error_shared[%d]: %f\n", tid, error_shared[tid]);
        oldCentroids[index] = temp;
    }

    // reduction step on the error array
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(tid < stride){
            error_shared[tid] += error_shared[tid + stride];
        }
    }
    __syncthreads();
    if(tid == 0){
        // add the error of the current block global error
        // printf("error: %f\n", *error);
        // printf("error_shared[0]: %f\n", error_shared[0]);

        atomicAdd(error, error_shared[0]);
        // printf("*********************: %f\n", error_shared[0]);
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

void cudaErrorCheck(cudaError_t error, string message) {
    // Check for kernel launch errors
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: %s in %s\n", cudaGetErrorString(error), message.c_str());
        // print status code
        fprintf(stderr, "Status code: %d\n", error);
        exit(-1);
    }
}
void kmeans(float * points, float * &centroids, int * &labels,  int nPoints, int nDimensions, int nCentroids, int maxIters){
    // Device Data
    float *d_points, *d_centroids, *d_oldCentroids, *error_val;
    int *d_labels, *d_counts;

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Allocate memory on GPU
    cudaMalloc(&d_points, nPoints * nDimensions * sizeof(float));
    // cudaErrorCheck("cudaMalloc d_points"
    cudaMalloc(&d_centroids, nCentroids * nDimensions * sizeof(float));
    // cudaErrorCheck("cudaMalloc d_centroids");
    cudaMalloc(&d_oldCentroids, nCentroids * nDimensions * sizeof(float));
    // cudaErrorCheck("cudaMalloc d_oldCentroids");
    cudaMalloc(&error_val, sizeof(float));
    // cudaErrorCheck("cudaMalloc error_val");
    cudaMalloc(&d_labels, nPoints * sizeof(int));
    // cudaErrorCheck("cudaMalloc d_labels");
    cudaMalloc(&d_counts, nCentroids * sizeof(int));
    // cudaErrorCheck("cudaMalloc d_counts");

    // Copy data to GPU
    cudaMemcpy(d_points, points, nPoints * nDimensions * sizeof(float), cudaMemcpyHostToDevice);
    // cudaErrorCheck("cudaMemcpy d_points");
    cudaMemcpy(d_centroids, centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyHostToDevice);
    // cudaErrorCheck("cudaMemcpy d_centroids");
    cudaMemcpy(d_oldCentroids, centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyHostToDevice);
    // cudaErrorCheck("cudaMemcpy d_oldCentroids");

    // Launch Kernel
    int labelingThreadsPerBlock = LABELING_BLOCK_SIZE;
    int labelingBlocksPerGrid = (nPoints + labelingThreadsPerBlock - 1) / labelingThreadsPerBlock;

    int updateThreadsPerBlock = UPDATE_BLOCK_SIZE;
    int updateBlocksPerGrid = (nCentroids * nDimensions + updateThreadsPerBlock - 1) / updateThreadsPerBlock;

    for(int i = 0; i < maxIters; i++){
        // initialize counts to 0
        cudaMemset(d_counts, 0, nCentroids * sizeof(int));
        cudaMemset(error_val, 0, sizeof(float));
        cudaMemset(d_centroids, 0, nCentroids * nDimensions * sizeof(float));
        // cudaErrorCheck("cudaMemset d_counts");

        // printf("Iteration %d\n", i);
        labelingKernel<<<labelingBlocksPerGrid, labelingThreadsPerBlock, nCentroids* nDimensions * sizeof(float)>>>(d_points, d_centroids, d_oldCentroids, d_labels, d_counts, nPoints, nDimensions, nCentroids);

        cudaErrorCheck(cudaDeviceSynchronize(),"labelingKernel");

        // Update Centroids
        updateKernel<<<updateBlocksPerGrid, updateThreadsPerBlock>>>(d_centroids, d_counts, d_oldCentroids, error_val, nDimensions, nCentroids);
        cudaDeviceSynchronize();
        cudaErrorCheck(cudaDeviceSynchronize(),"updateCentroids");
        // printf("updated\n");
        float error;
        cudaMemcpy(&error, error_val, sizeof(float), cudaMemcpyDeviceToHost);
        cudaErrorCheck(cudaDeviceSynchronize(),"cudaMemcpy error");
        // printf("Error: %f\n", error);
        if(error < MAX_ERR){
            printf("Converged\n");
            break;
        }
    }
    cudaMemcpy(centroids, d_centroids, nCentroids * nDimensions * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaErrorCheck("cudaMemcpy centroids");
    cudaMemcpy(labels, d_labels, nPoints * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaErrorCheck("cudaMemcpy labels");
    printf("Done\n");

    // Free memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_oldCentroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(error_val);
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.1f s \n", time/1000);
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
    readMatrix(file, points, nPoints, nDimensions);
    int step = nPoints / nCentroids;
    for(int i = 0; i < nCentroids; i++){
        for(int j = 0; j < nDimensions; j++){
            centroids[i*nDimensions+j] = points[i*step+j];
        }
    }
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