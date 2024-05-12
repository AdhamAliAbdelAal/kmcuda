#include "./knn.h"

__global__ void bubbleSort(double* data, int* labels, int n, int dim, double* target, int sizeToSort)
{

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * sizeToSort;
    int end = start + sizeToSort < n ? start + sizeToSort : n;
    if (start >= n) {
        return;
    }
    for (int i = start; i < end; i++) {
        for (int j = i + 1; j < end; j++) {
            double dist1 = 0;
            double dist2 = 0;
            for (int l = 0; l < dim; l++) {
                dist1 += (data[i * dim + l] - target[l]) * (data[i * dim + l] - target[l]);
                dist2 += (data[j * dim + l] - target[l]) * (data[j * dim + l] - target[l]);
            }
            if (dist1 > dist2) {
                for (int l = 0; l < dim; l++) {
                    double temp = data[i * dim + l];
                    data[i * dim + l] = data[j * dim + l];
                    data[j * dim + l] = temp;
                }
                int temp = labels[i];
                labels[i] = labels[j];
                labels[j] = temp;
            }
        }
    }
}

__device__ double euclidean_distance(double* data, int idx, double* target,
    int dim)
{
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (data[idx * dim + i] - target[i]) * (data[idx * dim + i] - target[i]);
    }
    return sqrt(sum);
}

__device__ unsigned int coRank(double* data, int dim, double* target, long long n, long long m, long long jOffset, long long iOffset, long long k)
{

    long long iLow = k > m ? k - m : 0;
    long long iHigh = n < k ? n : k;
    while (true) {
        long long i = (iHigh - iLow) / 2 + iLow;
        long long j = k - i;
        if (i > 0 && j < m && euclidean_distance(data, i + iOffset - 1, target, dim) > euclidean_distance(data, j + jOffset, target, dim)) {
            iHigh = i;

        } else if (j > 0 && i < n && euclidean_distance(data, j + jOffset - 1, target, dim) > euclidean_distance(data, i + iOffset, target, dim)) {
            iLow = i;
        } else {
            return i;
        }
    }
}

__device__ void sequintialMergeSegment(double* sharedDataA, double* sharedDataB, double* output, double* target, int n, int m, int dim)
{
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < n && j < m) {
        double dist1 = euclidean_distance(sharedDataA, i, target, dim);
        double dist2 = euclidean_distance(sharedDataB, j, target, dim);
        if (dist1 < dist2) {
            for (int l = 0; l < dim; l++) {
                output[k * dim + l] = sharedDataA[i * dim + l];
            }
            i++;
        } else {
            for (int l = 0; l < dim; l++) {
                output[k * dim + l] = sharedDataB[j * dim + l];
            }
            j++;
        }
        k++;
    }

    while (i < n) {
        for (int l = 0; l < dim; l++) {
            output[k * dim + l] = sharedDataA[i * dim + l];
        }
        i++;
        k++;
    }

    while (j < m) {
        for (int l = 0; l < dim; l++) {
            output[k * dim + l] = sharedDataB[j * dim + l];
        }
        j++;
        k++;
    }
}

__global__ void printArr(double* data, int n, int dim, double* target)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%f ", data[i * dim + j]);
        }

        // print distance
        double dist = 0;
        for (int j = 0; j < dim; j++) {
            dist += (data[i * dim + j] - target[j]) * (data[i * dim + j] - target[j]);
        }
        printf("Distance: %f\n", sqrt(dist));
    }
}

__global__ void mergeSort(double* data, int* labels, double* target, long long n, int dim, long long threadSize, long long sortedSize)
{
    threadSize = threadSize > sortedSize * 2 ? sortedSize * 2 : threadSize;

    // long long elementsPerBlock = blockDim.x * threadSize;
    long long elementsPerBlock = sortedSize * 2 > n ? n : sortedSize * 2;
    long long startThread = blockIdx.x * elementsPerBlock + threadIdx.x * threadSize;
    long long numSegment = (startThread) / sortedSize;
    long long jOffset = (numSegment + 1) * sortedSize;
    long long iOffset = (numSegment)*sortedSize;
    if (numSegment >= n || numSegment % 2 == 1) {
        return;
    }
    // elementsPerBlock = elementsPerBlock < n ? elementsPerBlock : n;
    long long kBlock = elementsPerBlock * blockIdx.x;
    long long kNextBlock = (kBlock + elementsPerBlock < n) ? (kBlock + elementsPerBlock) : n;

    if (kBlock >= n) {
        return;
    }
    // load data to shared memory
    __shared__ long long iBlock;
    __shared__ long long iNextBlock;
    __shared__ long long jBlock;
    __shared__ long long jNextBlock;
    if (threadIdx.x == 0) {
        long long jSize = n - jOffset < sortedSize ? n - jOffset : sortedSize;
        iBlock = coRank(data, sortedSize, target, sortedSize, sortedSize, jOffset, iOffset, kBlock);
        printf("==========================\n");
        printf("kBlock = %lld\n", kBlock);
        printf("kNextBlock = %lld\n", kNextBlock);
        printf("elementsPerBlock = %lld\n", elementsPerBlock);
        iNextBlock = coRank(data, sortedSize, target, sortedSize, sortedSize, jOffset, iOffset, kNextBlock);
        jBlock = kBlock - iBlock + jOffset;
        jNextBlock = kNextBlock - iNextBlock + jOffset;
    }

    __syncthreads();

    extern __shared__ double sharedArr[];
    // load data to shared memory
    long long nBlock = iNextBlock - iBlock;
    for (long long i = threadIdx.x; i < nBlock; i += blockDim.x) {
        for (int j = 0; j < dim; j++) {
            sharedArr[i * dim + j] = data[(i + iBlock) * dim + j];
        }
    }
    long long mBlock = jNextBlock - jBlock;
    for (long long i = threadIdx.x; i < mBlock; i += blockDim.x) {
        for (int j = 0; j < dim; j++) {
            sharedArr[(i + nBlock) * dim + j] = data[(i + jBlock) * dim + j];
        }
    }

    // if (numSegment % 2 == 1){

    // }

    __syncthreads();
    if (threadIdx.x == 0) {
        // printf("threadSize = %lld sortedSize = %lld elementsPerBlock = %lld\n", threadSize, sortedSize, elementsPerBlock);
        // printf("elementsPerBlock = %lld\n", elementsPerBlock);
        // printf("threadIdx.x = %lld nBlock: %lld, mBlock: %lld\n", threadIdx.x, nBlock, mBlock);
        // printf("threadIdx.x = %lld iBlock: %lld, iNextBlock: %lld\n", threadIdx.x, iBlock, iNextBlock);
        // printf("threadIdx.x = %lld jBlock: %lld, jNextBlock: %lld\n", threadIdx.x, jBlock, jNextBlock);
        // printf("threadIdx.x = %lld kBlock: %lld, kNextBlock: %lld\n", threadIdx.x, kBlock, kNextBlock);
        // printf("nBlock = %lld, mBlock = %lld\n", nBlock, mBlock);
        // print sharedArr
        // for (int i = 0; i < nBlock; i++) {
        //     for (int j = 0; j < dim; j++) {
        //         printf("%f ", sharedArr[i * dim + j]);
        //     }
        //     // print distance
        //     double dist = 0;
        //     for (int j = 0; j < dim; j++) {
        //         dist += (sharedArr[i * dim + j] - target[j]) * (sharedArr[i * dim + j] - target[j]);
        //     }
        //     printf("Distance: %f\n", sqrt(dist));
        // }
        // printf("arr b\n");

        // for (int i = 0; i < mBlock; i++) {
        //     for (int j = 0; j < dim; j++) {
        //         printf("%f ", sharedArr[(i + nBlock) * dim + j]);
        //     }
        //     // print distance
        //     double dist = 0;
        //     for (int j = 0; j < dim; j++) {
        //         dist += (sharedArr[(i + nBlock) * dim + j] - target[j]) * (sharedArr[(i + nBlock) * dim + j] - target[j]);
        //     }
        //     printf("Distance: %f\n", sqrt(dist));
        // }
    }

    // shared memory for output
    double* output = sharedArr + elementsPerBlock * dim;

    long long k = threadIdx.x * threadSize;
    if (k < elementsPerBlock) {
        long long i = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, k);
        long long j = k - i + nBlock;
        long long kNext = k + threadSize < elementsPerBlock ? k + threadSize : elementsPerBlock;
        long long iNext = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, kNext);
        long long jNext = kNext - iNext + nBlock;

        // printf("threadIdx.x = %d k: %lld, i: %lld, iNext: %lld\n", threadIdx.x, k, i, iNext);
        // printf("threadIdx.x = %d k: %lld, j: %lld, jNext: %lld\n", threadIdx.x, k, j, jNext);

        // sequential merge
        long long n = iNext - i;
        long long m = jNext - j;
        sequintialMergeSegment(sharedArr + i * dim, sharedArr + j * dim, output + k * dim, target, n, m, dim);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        // print output
        // for (int i = 0; i < elementsPerBlock; i++) {
        //     for (int j = 0; j < dim; j++) {
        //         printf("%f ", output[i * dim + j]);
        //     }
        //     printf("\n");
        // }
    }

    // copy output to data
    for (long long i = threadIdx.x; i < elementsPerBlock; i += blockDim.x) {
        for (int j = 0; j < dim; j++) {
            data[(i + kBlock) * dim + j] = output[i * dim + j];
        }
    }
    if (threadIdx.x == 0) {
        // print data
        // printf("data = \n");
        // for (int i = 0; i < elementsPerBlock; i++) {
        //     for (int j = 0; j < dim; j++) {
        //         printf("%f ", data[(i + kBlock) * dim + j]);
        //     }
        //     // print distance
        //     double dist = 0;
        //     for (int j = 0; j < dim; j++) {
        //         dist += (data[(i + kBlock) * dim + j] - target[j]) * (data[(i + kBlock) * dim + j] - target[j]);
        //     }
        //     printf("Distance: %f\n", sqrt(dist));
        // }
    }
    __syncthreads();
}
