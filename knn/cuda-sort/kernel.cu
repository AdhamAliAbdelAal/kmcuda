#include "./knn.h"

__global__ void bubbleSort(float *data, int *labels, int n, int dim, float *target, int sizeToSort)
{

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * sizeToSort;
    int end = start + sizeToSort < n ? start + sizeToSort : n;
    if (start >= n)
    {
        return;
    }
    for (int i = start; i < end; i++)
    {
        for (int j = i + 1; j < end; j++)
        {
            float dist1 = 0;
            float dist2 = 0;
            for (int l = 0; l < dim; l++)
            {
                dist1 += (data[i * dim + l] - target[l]) * (data[i * dim + l] - target[l]);
                dist2 += (data[j * dim + l] - target[l]) * (data[j * dim + l] - target[l]);
            }
            if (dist1 > dist2)
            {
                for (int l = 0; l < dim; l++)
                {
                    float temp = data[i * dim + l];
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

__device__ float euclidean_distance(float *data, int idx, float *target, int dim)
{
    float sum = 0;
    for (int i = 0; i < dim; i++)
    {
        sum += (data[idx * dim + i] - target[i]) * (data[idx * dim + i] - target[i]);
    }
    return sqrt(sum);
}

__device__ unsigned int coRank(float *data, int dim, float *target, long long n, long long m, long long jOffset,
                               long long iOffset, long long k)
{

    long long iLow = max(k - m, iOffset - 1);
    long long iHigh = min(n + iOffset, k + 1);
    int tttt = 0;
    while (true)
    {
        long long i = (iHigh - iLow) / 2 + iLow;
        long long j = k - i + jOffset;
        tttt++;
        if (tttt == 10000)
        {
            printf("stuck\n");
            printf("stuck blockIdx.x %d threadIdx.x = %d i: %lld, j: %lld, iLow: %lld, iHigh: %lld\n", blockIdx.x,
                   threadIdx.x, i, j, iLow, iHigh);
            // return i;
        }
        if (i > iOffset && j < m + jOffset &&
            euclidean_distance(data, i - 1, target, dim) > euclidean_distance(data, j, target, dim))
        {
            iHigh = i;
        }
        else if (j > jOffset && i < n + iOffset &&
                 euclidean_distance(data, j - 1, target, dim) > euclidean_distance(data, i, target, dim))
        {
            iLow = i;
        }
        else
        {
            return i;
        }
    }
}

__device__ void sequintialMergeSegment(float *sharedDataA, float *sharedDataB, float *output, float *target, int n,
                                       int m, int dim)
{
    int i = 0;
    int j = 0;
    int k = 0;
    while (i < n && j < m)
    {
        float dist1 = euclidean_distance(sharedDataA, i, target, dim);
        float dist2 = euclidean_distance(sharedDataB, j, target, dim);
        if (dist1 < dist2)
        {
            for (int l = 0; l < dim; l++)
            {
                output[k * dim + l] = sharedDataA[i * dim + l];
            }
            i++;
        }
        else
        {
            for (int l = 0; l < dim; l++)
            {
                output[k * dim + l] = sharedDataB[j * dim + l];
            }
            j++;
        }
        k++;
    }

    while (i < n)
    {
        for (int l = 0; l < dim; l++)
        {
            output[k * dim + l] = sharedDataA[i * dim + l];
        }
        i++;
        k++;
    }

    while (j < m)
    {
        for (int l = 0; l < dim; l++)
        {
            output[k * dim + l] = sharedDataB[j * dim + l];
        }
        j++;
        k++;
    }
}

__global__ void printArr(float *data, int n, int dim, float *target)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf("%f ", data[i * dim + j]);
        }

        // print distance
        float dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (data[i * dim + j] - target[j]) * (data[i * dim + j] - target[j]);
        }
        printf("Distance: %f\n", sqrt(dist));
    }
}

__global__ void mergeSort(float *data, int *labels, float *target, long long n, int dim, long long threadSize,
                          long long sortedSize)
{
    threadSize = threadSize > sortedSize * 2 ? sortedSize * 2 : threadSize;

    long long maxElementsPerBlock = blockDim.x * threadSize > sortedSize * 2 ? sortedSize * 2 : blockDim.x * threadSize;
    long long elementsPerBlock = maxElementsPerBlock < n ? maxElementsPerBlock : n;

    // load data to shared memory
    __shared__ long long iBlock;
    __shared__ long long iNextBlock;
    __shared__ long long jBlock;
    __shared__ long long jNextBlock;
    __shared__ long long kBlock;
    __shared__ long long kNextBlock;
    __shared__ long long jOffset;
    __shared__ long long iOffset;

    if (threadIdx.x == 0)
    {
        long long numBlocksPerSortedSize = (2 * sortedSize + maxElementsPerBlock - 1) / maxElementsPerBlock;
        long long whichSortedSize = blockIdx.x / numBlocksPerSortedSize;
        long long startSortedSize = whichSortedSize * sortedSize * 2;
        long long blockOrderInSortedSize = blockIdx.x % numBlocksPerSortedSize;

        kBlock = startSortedSize + blockOrderInSortedSize * maxElementsPerBlock;
        kNextBlock = min(kBlock + maxElementsPerBlock, min(n, startSortedSize + 2 * sortedSize));
        iOffset = whichSortedSize * sortedSize * 2;
        jOffset = iOffset + sortedSize;
        if (kBlock < n)
        {
            printf(
                "blockIdx.x: %d, whichSortedSize: %lld, startSortedSize: %lld, blockOrderInSortedSize: %lld, kBlock: "
                "%lld, kNextBlock: %lld kBlock-kNextBlock: %lld\n",
                blockIdx.x, whichSortedSize, startSortedSize, blockOrderInSortedSize, kBlock, kNextBlock,
                kNextBlock - kBlock);
            long long jSize = n - jOffset < sortedSize ? n - jOffset : sortedSize;
            long long iSize = n - iOffset < sortedSize ? n - iOffset : sortedSize;
            if (jSize < 0)
            {
                jSize = 0;
            }
            iBlock = coRank(data, dim, target, iSize, jSize, jOffset, iOffset, kBlock);
            iNextBlock = coRank(data, dim, target, iSize, jSize, jOffset, iOffset, kNextBlock);

            jBlock = kBlock - iBlock + jOffset;
            jNextBlock = kNextBlock - iNextBlock + jOffset;
        }
    }

    __syncthreads();
    return;
    long long nBlock, mBlock;
    extern __shared__ float sharedArr[];
    if (kBlock < n)
    {
        // load data to shared memory
        nBlock = iNextBlock - iBlock;
        for (long long i = threadIdx.x; i < nBlock; i += blockDim.x)
        {

            for (int j = 0; j < dim; j++)
            {
                sharedArr[i * dim + j] = data[(i + iBlock) * dim + j];
            }
        }
        mBlock = jNextBlock - jBlock;
        if (threadIdx.x == 0)
        {
            printf("nBlock: %lld, mBlock: %lld maxElementsPerBlock: %lld\n", nBlock, mBlock, maxElementsPerBlock);
        }
        for (long long i = threadIdx.x; i < mBlock; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                sharedArr[(i + nBlock) * dim + j] = data[(i + jBlock) * dim + j];
            }
        }
    }

    __syncthreads();

    long long actualSize = nBlock + mBlock;
    // shared memory for output
    float *output = sharedArr + (actualSize)*dim;

    long long k = threadIdx.x * threadSize;
    if (kBlock < n && k < actualSize)
    {
        long long i = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, k);
        long long j = k - i + nBlock;
        long long kNext = k + threadSize < actualSize ? k + threadSize : actualSize;
        long long iNext = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, kNext);
        long long jNext = kNext - iNext + nBlock;

        // sequential merge
        long long n = iNext - i;
        long long m = jNext - j;
        sequintialMergeSegment(sharedArr + i * dim, sharedArr + j * dim, output + k * dim, target, n, m, dim);
    }
    __syncthreads();

    if (kBlock < n)
    {
        // copy output to data
        for (long long i = threadIdx.x; i < elementsPerBlock; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                data[(i + kBlock) * dim + j] = output[i * dim + j];
            }
        }
    }
}
