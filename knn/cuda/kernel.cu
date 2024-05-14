#include <iostream>

__global__ void calcDistances(double *data, double *target, double *distances, long long n, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        {
            double sum = 0;
            for (int j = 0; j < dim; j++)
            {
                sum += (data[i * dim + j] - target[j]) * (data[i * dim + j] - target[j]);
            }
            distances[i] = sqrt(sum);
        }
    }
}

__device__ double euclidean_distance(double *data, int idx, double *target, int dim)
{
    double sum = 0;
    for (int i = 0; i < dim; i++)
    {
        sum += (data[idx * dim + i] - target[i]) * (data[idx * dim + i] - target[i]);
    }
    return sqrt(sum);
}

__device__ void sortElements(double *data, double *distances, double *distancesOut, long long startElement,
                             long long endElement, long long startElementCopy, long long endElementCopy, double *target,
                             int k, int *nearesrtNeighborsIdxs, int dim)
{
    // Initialize nearestNeighborsIdxs and distancesOut with initial k values
    for (int i = 0; i < k; i++)
    {
        nearesrtNeighborsIdxs[i] = startElement + i;
        distancesOut[i + startElementCopy] = distances[startElement + i];
    }

    // Sort distances and update nearestNeighborsIdxs
    for (long long i = startElement + k; i <= endElement; i++)
    {
        // Find the index of the maximum distance in distancesOut
        long long maxIdx = 0;
        double maxVal = distancesOut[startElementCopy];
        for (long long j = 1; j < k; j++)
        {
            if (distancesOut[j + startElementCopy] > maxVal)
            {
                maxIdx = j;
                maxVal = distancesOut[j + startElementCopy];
            }
        }

        // Update distancesOut and nearestNeighborsIdxs if the current distance is smaller
        if (distances[i] < maxVal)
        {
            distancesOut[maxIdx + startElementCopy] = distances[i];
            nearesrtNeighborsIdxs[maxIdx] = i;
        }
    }
}

__global__ void knn(double *data, int *labels, double *distances, int threadSize, long long n, int dim, int k,
                    double *target, double *output, int *labelsOutput, double *distancesOut)
{
    // print thread info
    long long startElement = (blockIdx.x * blockDim.x + threadIdx.x) * threadSize;
    long long endElement = startElement + threadSize < n ? startElement + threadSize - 1 : n - 1;
    long long totalElements = endElement - startElement + 1;
    k = k < totalElements ? k : totalElements;

    if (startElement < n)
    {
        // copy elements to data again
        long long startElementCopy = (blockIdx.x * blockDim.x + threadIdx.x) * k;
        long long endElementCopy = startElementCopy + k - 1;

        int *nearesrtNeighborsIdxs = (int *)malloc(sizeof(int) * k);
        sortElements(data, distances, distancesOut, startElement, endElement, startElementCopy, endElementCopy, target,
                     k, nearesrtNeighborsIdxs, dim);

        for (long long i = startElementCopy; i <= endElementCopy; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                output[i * dim + j] = data[nearesrtNeighborsIdxs[i - startElementCopy] * dim + j];
            }
            labelsOutput[i] = labels[nearesrtNeighborsIdxs[i - startElementCopy]];
        }
        free(nearesrtNeighborsIdxs);
    }
}