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

    long long iLow = k + iOffset > m ? k + iOffset - m : iOffset;
    long long iHigh = n < k ? n : k;
    while (true) {
        long long i = (iHigh - iLow) / 2 + iLow;
        long long j = k - i;
        if (i > iOffset && j < m && euclidean_distance(data, i - 1, target, dim) > euclidean_distance(data, j, target, dim)) {
            iHigh = i;

        } else if (j > jOffset && i < n && euclidean_distance(data, j - 1, target, dim) > euclidean_distance(data, i, target, dim)) {
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

    long long elementsPerBlock = blockDim.x * threadSize;
    // long long elementsPerBlock = sortedSize * 2 > n ? n : sortedSize * 2;

    // elementsPerBlock = elementsPerBlock < n ? elementsPerBlock : n;
    long long kBlock = elementsPerBlock * blockIdx.x;
    long long kNextBlock = (kBlock + elementsPerBlock < n) ? (kBlock + elementsPerBlock) : n;

    // load data to shared memory
    __shared__ long long iBlock;
    __shared__ long long iNextBlock;
    __shared__ long long jBlock;
    __shared__ long long jNextBlock;
    __shared__ long long blockStart;
    __shared__ long long numSegment;
    __shared__ long long jOffset;
    __shared__ long long iOffset;

    if (threadIdx.x == 0) {
        blockStart = blockIdx.x * elementsPerBlock;
        numSegment = ((blockStart) / (sortedSize * 2)) * 2;
        jOffset = (numSegment + 1) * sortedSize;
        iOffset = (numSegment)*sortedSize;

        printf("blockIdx.x = %d blockStart: %lld\n", blockIdx.x, blockStart);
        printf("blockIdx.x = %d numSegment: %lld\n", blockIdx.x, numSegment);
        printf("blockIdx.x = %d jOffset: %lld\n", blockIdx.x, jOffset);
        printf("blockIdx.x = %d iOffset: %lld\n", blockIdx.x, iOffset);

        long long jSize = n - jOffset < sortedSize ? n - jOffset : sortedSize;
        iBlock = coRank(data, sortedSize, target, sortedSize, jSize, jOffset, iOffset, kBlock);
        iNextBlock = coRank(data, sortedSize, target, sortedSize, jSize, jOffset, iOffset, kNextBlock);

        jBlock = kBlock - iBlock + jOffset;
        jNextBlock = kNextBlock - iNextBlock + jOffset;

        printf("blockIdx.x = %d iBlock: %lld\n", blockIdx.x, iBlock);
        printf("blockIdx.x = %d iNextBlock: %lld\n", blockIdx.x, iNextBlock);
        printf("blockIdx.x = %d jBlock: %lld\n", blockIdx.x, jBlock);
        printf("blockIdx.x = %d jNextBlock: %lld\n", blockIdx.x, jNextBlock);
        printf("blockIdx.x = %d kBlock: %lld\n", blockIdx.x, kBlock);
        printf("blockIdx.x = %d jSize: %lld\n", blockIdx.x, jSize);
    }

    __syncthreads();
    long long nBlock, mBlock;
    extern __shared__ double sharedArr[];
    if (numSegment < n) {

        // load data to shared memory
        nBlock = iNextBlock - iBlock;
        for (long long i = threadIdx.x; i < nBlock; i += blockDim.x) {
            for (int j = 0; j < dim; j++) {
                sharedArr[i * dim + j] = data[(i + iBlock) * dim + j];
            }
        }
        mBlock = jNextBlock - jBlock;
        for (long long i = threadIdx.x; i < mBlock; i += blockDim.x) {
            for (int j = 0; j < dim; j++) {
                sharedArr[(i + nBlock) * dim + j] = data[(i + jBlock) * dim + j];
            }
        }

        // print shared memory
        if (threadIdx.x == 0) {
            printf("shared memory\n");
            for (int i = 0; i < nBlock + mBlock; i++) {
                for (int j = 0; j < dim; j++) {
                    printf("blockIdx.x:%d %f ", blockIdx.x, sharedArr[i * dim + j]);
                }
                printf("\n");
            }
        }
    }

    __syncthreads();

    // shared memory for output
    double* output = sharedArr + elementsPerBlock * dim;

    long long k = threadIdx.x * threadSize;
    if (numSegment < n) {
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
    }
    __syncthreads();

    if (numSegment < n || numSegment % 2 != 1) {
        // copy output to data
        for (long long i = threadIdx.x; i < elementsPerBlock; i += blockDim.x) {
            for (int j = 0; j < dim; j++) {
                data[(i + kBlock) * dim + j] = output[i * dim + j];
            }
        }
    }
    __syncthreads();
}
