#include <iostream>
#include <cmath>
using namespace std;

#define MAX_ERR 1e-6


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

float euclideanDistance(float *point, float *centroid, int nDimensions){
    float distance = 0;
    for(int i = 0; i < nDimensions; i++){
        distance += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return distance;
}

void assignLabels(float *points, float *centroids, int *labels, int nPoints, int nDimensions, int nCentroids){
    for(int i = 0; i < nPoints; i++){
        float minDistance = euclideanDistance(points + i*nDimensions, centroids, nDimensions);
        labels[i] = 0;
        for(int j = 1; j < nCentroids; j++){
            float distance = euclideanDistance(points + i*nDimensions, centroids + j*nDimensions, nDimensions);
            if (distance < minDistance){
                minDistance = distance;
                labels[i] = j;
            }
        }
    }
}

void updateCentroids(float *points, float *centroids, int *labels, int nPoints, int nDimensions, int nCentroids){
    int *count = (int*)malloc(nCentroids * sizeof(int));
    for(int i = 0; i < nCentroids; i++){
        count[i] = 0;
        for(int j = 0; j < nDimensions; j++){
            centroids[i*nDimensions+j] = 0;
        }
    }
    for(int i = 0; i < nPoints; i++){
        int label = labels[i];
        count[label]++;
        for(int j = 0; j < nDimensions; j++){
            centroids[label*nDimensions+j] += points[i*nDimensions+j];
        }
    }
    for(int i = 0; i < nCentroids; i++){
        for(int j = 0; j < nDimensions; j++){
            centroids[i*nDimensions+j] /= count[i];
        }
    }
    free(count);
}

float getError(float *oldCentroids, float *centroids, int nDimensions, int nCentroids){
    float error = 0;
    for(int i = 0; i < nCentroids; i++){
        for(int j = 0; j < nDimensions; j++){
            error += abs(oldCentroids[i*nDimensions+j] - centroids[i*nDimensions+j]);
        }
    }
    return error;
}
void kmeans(float *points, float *centroids, int *labels, int nPoints, int nDimensions, int nCentroids, int maxIters){
    float* oldCentroids = allocateMatrix(nCentroids, nDimensions);
    for(int j = 0; j < nCentroids; j++){
        for(int k = 0; k < nDimensions; k++){
            oldCentroids[j*nDimensions+k] = centroids[j*nDimensions+k];
        }
    }
    for(int i=0;i<maxIters;i++){
        printf("Iteration %d\n", i);
        assignLabels(points, centroids, labels, nPoints, nDimensions, nCentroids);
        // printf("Assign labels Done\n");
        updateCentroids(points, centroids, labels, nPoints, nDimensions, nCentroids);
        float error = getError(oldCentroids, centroids, nDimensions, nCentroids);
        printf("Error = %f\n", error);
        if (error < MAX_ERR){
            break;
        }
        for(int j = 0; j < nCentroids; j++){
            for(int k = 0; k < nDimensions; k++){
                oldCentroids[j*nDimensions+k] = centroids[j*nDimensions+k];
            }
        }
    }
    freeMatrix(oldCentroids);
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