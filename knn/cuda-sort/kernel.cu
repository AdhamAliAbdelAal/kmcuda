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

__device__ long long coRank(float *data, int dim, float *target, long long n, long long m, long long jOffset,
                            long long iOffset, long long k)
{

    long long iLow = max(k - m, iOffset);
    long long iHigh = min(n + iOffset, k);
    // int tttt = 0;
    while (true)
    {
        long long i = (iHigh - iLow) / 2 + iLow;
        long long j = k - i + jOffset;
        bool takeAction = false;
        if (iLow == iHigh - 1)
        {
            takeAction = true;
        }
        // tttt++;
        // if (tttt == 10000)
        // {

        //     printf("stuck blockIdx.x %d threadIdx.x = %d i: %lld, j: %lld, iLow: %lld, iHigh: %lld\n", blockIdx.x,
        //            threadIdx.x, i, j, iLow, iHigh);
        //     printf("stuck threadIdx.x = %d max(k - m, iOffset): %lld, min(n + iOffset, k): %lld\n", threadIdx.x,
        //            max(k - m, iOffset), min(n + iOffset, k));
        //     printf("stuck jOffset: %lld, iOffset: %lld, k: %lld\n", jOffset, iOffset, k);
        //     printf("stuck n: %lld, m: %lld\n", n, m);
        //     // return i;
        // }
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
        if (iLow == iHigh - 1 && takeAction)
        {
            return iHigh;
        }
    }
}

__device__ void sequintialMergeSegment(float *sharedDataA, float *sharedDataB, float *output, float *target,
                                       long long n, long long m, int dim)
{
    long long i = 0;
    long long j = 0;
    long long k = 0;
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
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("done-1\n");
    threadSize = threadSize > sortedSize * 2 ? sortedSize * 2 : threadSize;

    long long maxElementsPerBlock = blockDim.x * threadSize > sortedSize * 2 ? sortedSize * 2 : blockDim.x * threadSize;

    // load data to shared memory
    __shared__ long long iBlock;
    __shared__ long long iNextBlock;
    __shared__ long long jBlock;
    __shared__ long long jNextBlock;
    __shared__ long long kBlock;
    __shared__ long long kNextBlock;
    __shared__ long long jOffset;
    __shared__ long long iOffset;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("done0\n");
    if (threadIdx.x == 0)
    {
        long long numBlocksPerSortedSize = (2 * sortedSize + maxElementsPerBlock - 1) / maxElementsPerBlock;
        long long whichSortedSize = blockIdx.x / numBlocksPerSortedSize;
        long long startSortedSize = whichSortedSize * sortedSize * 2;
        long long blockOrderInSortedSize = blockIdx.x % numBlocksPerSortedSize;

        kBlock = startSortedSize + blockOrderInSortedSize * maxElementsPerBlock;
        kNextBlock = min(kBlock + maxElementsPerBlock, min(n, startSortedSize + 2 * sortedSize));
        iOffset = startSortedSize;
        jOffset = startSortedSize + sortedSize;
        if (kBlock < n)
        {
            long long jSize = min(n - jOffset, sortedSize);
            long long iSize = min(n - iOffset, sortedSize);
            if (jSize < 0)
            {
                jSize = 0;
            }
            if (iSize < 0)
            {
                iSize = 0;
            }
            iBlock = coRank(data, dim, target, iSize, jSize, jOffset, iOffset, kBlock);
            iNextBlock = coRank(data, dim, target, iSize, jSize, jOffset, iOffset, kNextBlock);

            jBlock = kBlock - iBlock + jOffset;
            jNextBlock = kNextBlock - iNextBlock + jOffset;
        }
    }

    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("done1\n");

    long long nBlock = 0, mBlock = 0;
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
        for (long long i = threadIdx.x; i < mBlock; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                sharedArr[(i + nBlock) * dim + j] = data[(i + jBlock) * dim + j];
            }
        }
    }

    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)

        printf("done2\n");

    long long actualSize = nBlock + mBlock;
    // shared memory for output
    float *output = sharedArr + (actualSize)*dim;

    long long k = threadIdx.x * threadSize;
    if (kBlock < n && k < actualSize)
    {
        // printf("actualSize: %lld, k: %lld nBlock: %lld, mBlock: %lld\n", actualSize, k, nBlock, mBlock);
        long long i = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, k);
        long long j = k - i + nBlock;
        long long kNext = min(k + threadSize, actualSize);
        long long iNext = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, kNext);
        long long jNext = kNext - iNext + nBlock;

        // sequential merge
        long long nn = iNext - i;
        long long mm = jNext - j;

        // printf("k: %lld, i: %lld, j: %lld, actualSize: %lld, mm: %lld, nn: %lld\n", k, i, j, actualSize, mm, nn);
        sequintialMergeSegment(sharedArr + i * dim, sharedArr + j * dim, output + k * dim, target, nn, mm, dim);
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("done3\n");
    if (kBlock < n)
    {
        // copy output to data
        for (long long i = threadIdx.x; i < actualSize; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                data[(i + kBlock) * dim + j] = output[i * dim + j];
            }
        }
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("done4\n");
}
