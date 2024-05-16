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
                dist1 += (data[i + l * n] - target[l]) * (data[i + l * n] - target[l]);
                dist2 += (data[j + l * n] - target[l]) * (data[j + l * n] - target[l]);
            }
            if (dist1 > dist2)
            {
                for (int l = 0; l < dim; l++)
                {
                    float temp = data[i + l * n];
                    data[i + l * n] = data[j + l * n];
                    data[j + l * n] = temp;
                }
                int temp = labels[i];
                labels[i] = labels[j];
                labels[j] = temp;
            }
        }
    }
}

__device__ float euclidean_distance(float *data, int idx, float *target, int dim, long long n)
{
    float sum = 0;
    for (int i = 0; i < dim; i++)
    {
        sum += (data[idx + i * n] - target[i]) * (data[idx + i * n] - target[i]);
    }
    return sqrt(sum);
}

__device__ long long coRank(float *data, int dim, float *target, long long n, long long m, long long jOffset,
                            long long iOffset, long long k, long long dataN)
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
            euclidean_distance(data, i - 1, target, dim, dataN) > euclidean_distance(data, j, target, dim, dataN))
        {

            iHigh = i;
        }
        else if (j > jOffset && i < n + iOffset &&
                 euclidean_distance(data, j - 1, target, dim, dataN) > euclidean_distance(data, i, target, dim, dataN))
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
                                       long long n, long long m, int dim, long long actualSize)
{
    long long i = 0;
    long long j = 0;
    long long k = 0;
    while (i < n && j < m)
    {
        float dist1 = euclidean_distance(sharedDataA, i, target, dim, actualSize);
        float dist2 = euclidean_distance(sharedDataB, j, target, dim, actualSize);
        if (dist1 < dist2)
        {
            for (int l = 0; l < dim; l++)
            {
                output[k + l * actualSize] = sharedDataA[i + l * actualSize];
            }
            i++;
        }
        else
        {
            for (int l = 0; l < dim; l++)
            {
                output[k + l * actualSize] = sharedDataB[j + l * actualSize];
            }
            j++;
        }
        k++;
    }

    while (i < n)
    {
        for (int l = 0; l < dim; l++)
        {
            output[k + l * actualSize] = sharedDataA[i + l * actualSize];
        }
        i++;
        k++;
    }

    while (j < m)
    {
        for (int l = 0; l < dim; l++)
        {
            output[k + l * actualSize] = sharedDataB[j + l * actualSize];
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
            printf("%f ", data[i + j * n]);
        }

        // print distance
        float dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (data[i + j * n] - target[j]) * (data[i + j * n] - target[j]);
        }
        printf("Distance: %f\n", sqrt(dist));
    }
}

__global__ void mergeSort(float *data, int *labels, float *target, long long n, int dim, long long threadSize,
                          long long sortedSize)
{
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("done-1\n");
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
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("done0\n");
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
            iBlock = coRank(data, dim, target, iSize, jSize, jOffset, iOffset, kBlock, n);
            iNextBlock = coRank(data, dim, target, iSize, jSize, jOffset, iOffset, kNextBlock, n);

            jBlock = kBlock - iBlock + jOffset;
            jNextBlock = kNextBlock - iNextBlock + jOffset;
        }
    }

    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("done1\n");

    long long actualSize = 0;
    long long nBlock = 0, mBlock = 0;
    extern __shared__ float sharedArr[];
    if (kBlock < n)
    {
        // load data to shared memory
        nBlock = iNextBlock - iBlock;
        mBlock = jNextBlock - jBlock;
        nBlock = min(maxElementsPerBlock, nBlock);
        mBlock = min(maxElementsPerBlock, mBlock);
        nBlock = max(0LL, nBlock);
        mBlock = max(0LL, mBlock);
        actualSize = nBlock + mBlock;
        if (nBlock < 0 || mBlock < 0)
        {
            printf("0 nBlock: %lld, mBlock: %lld\n", nBlock, mBlock);
        }
        if (nBlock > maxElementsPerBlock || mBlock > maxElementsPerBlock)
        {
            printf("max nBlock: %lld, mBlock: %lld, maxElementsPerBlock: %lld\n", nBlock, mBlock, maxElementsPerBlock);
        }
        if (nBlock + mBlock <= maxElementsPerBlock)
        {
            for (long long i = threadIdx.x; i < nBlock; i += blockDim.x)
            {
                for (int j = 0; j < dim; j++)
                {
                    sharedArr[i + j * actualSize] = data[(i + iBlock) + j * n];
                }
            }
            for (long long i = threadIdx.x; i < mBlock; i += blockDim.x)
            {
                for (int j = 0; j < dim; j++)
                {
                    sharedArr[(i + nBlock) + j * actualSize] = data[(i + jBlock) + j * n];
                }
            }
        }
        else
        {
            printf("errrrrrrrror\n");
            printf("nBlock: %lld, mBlock: %lld, maxElementsPerBlock: %lld\n", nBlock, mBlock, maxElementsPerBlock);
        }
    }

    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0)

    //     printf("done2\n");

    // shared memory for output
    float *output = sharedArr + (actualSize)*dim;

    long long k = threadIdx.x * threadSize;
    if (kBlock < n && k < actualSize)
    {
        // printf("actualSize: %lld, k: %lld nBlock: %lld, mBlock: %lld\n", actualSize, k, nBlock, mBlock);
        long long i = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, k, actualSize);
        long long j = k - i + nBlock;
        long long kNext = min(k + threadSize, actualSize);
        long long iNext = coRank(sharedArr, dim, target, nBlock, mBlock, nBlock, 0, kNext, actualSize);
        long long jNext = kNext - iNext + nBlock;

        // sequential merge
        long long nn = iNext - i;
        long long mm = jNext - j;

        // printf("k: %lld, i: %lld, j: %lld, actualSize: %lld, mm: %lld, nn: %lld\n", k, i, j, actualSize, mm, nn);
        sequintialMergeSegment(sharedArr + i, sharedArr + j, output + k, target, nn, mm, dim, actualSize);
    }
    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("done3\n");
    if (kBlock < n)
    {
        // copy output to data
        for (long long i = threadIdx.x; i < actualSize; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                data[(i + kBlock) + j * n] = output[i + j * actualSize];
            }
        }
    }
    // __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     printf("done4\n");
}
