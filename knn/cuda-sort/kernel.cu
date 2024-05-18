#include "./knn.h"

__global__ void calcDistances(float *data, float *target, float *distances, long long n, int dim, long long totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        {
            float sum = 0;
            for (int j = 0; j < dim; j++)
            {
                sum += (data[i + j * totalSize] - target[j]) * (data[i + j * totalSize] - target[j]);
            }
            distances[i] = sqrt(sum);
        }
    }
}

__global__ void bubbleSort(float *data, int *labels, float *distances, long long n, int dim, int sizeToSort,
                           long long totalSize)
{

    long long start = (blockIdx.x * blockDim.x + threadIdx.x) * sizeToSort;
    long long end = start + sizeToSort < n ? start + sizeToSort : n;
    if (start >= n)
    {
        return;
    }
    for (int i = start; i < end; i++)
    {
        for (int j = i + 1; j < end; j++)
        {
            float dist1 = distances[i];
            float dist2 = distances[j];
            if (dist1 > dist2)
            {
                for (int l = 0; l < dim; l++)
                {
                    float temp = data[i + l * totalSize];
                    data[i + l * totalSize] = data[j + l * totalSize];
                    data[j + l * totalSize] = temp;
                }
                int temp = labels[i];
                labels[i] = labels[j];
                labels[j] = temp;

                float tempDist = distances[i];
                distances[i] = distances[j];
                distances[j] = tempDist;
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

__device__ long long coRank(float *data, int dim, float *distances, long long n, long long m, long long jOffset,
                            long long iOffset, long long k, long long dataN)
{

    long long iLow = max(k - m, iOffset);
    long long iHigh = min(n + iOffset, k);
    while (true)
    {
        long long i = (iHigh - iLow) / 2 + iLow;
        long long j = k - i + jOffset;
        bool takeAction = false;
        if ((iLow == iHigh - 1 || iLow == iHigh))
        {
            takeAction = true;
        }
        if (i > iOffset && j < m + jOffset && distances[i - 1] > distances[j])
        {

            iHigh = i;
        }
        else if (j > jOffset && i < n + iOffset && distances[j - 1] > distances[i])
        {
            iLow = i;
        }
        else
        {
            return i;
        }
        if ((iLow == iHigh - 1 || iLow == iHigh) && takeAction)
        {
            return iHigh;
        }
    }
}

__device__ void sequintialMergeSegment(float *sharedDataA, float *sharedDataB, float *output, float *distancesA,
                                       float *distancesB, float *outputDistances, long long n, long long m, int dim,
                                       long long actualSize)
{
    long long i = 0;
    long long j = 0;
    long long k = 0;
    while (i < n && j < m)
    {
        float dist1 = distancesA[i];
        float dist2 = distancesB[j];
        if (dist1 < dist2)
        {
            for (int l = 0; l < dim; l++)
            {
                output[k + l * actualSize] = sharedDataA[i + l * actualSize];
            }
            outputDistances[k] = dist1;
            i++;
        }
        else
        {
            for (int l = 0; l < dim; l++)
            {
                output[k + l * actualSize] = sharedDataB[j + l * actualSize];
            }
            outputDistances[k] = dist2;
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
        outputDistances[k] = distancesA[i];
        i++;
        k++;
    }

    while (j < m)
    {
        for (int l = 0; l < dim; l++)
        {
            output[k + l * actualSize] = sharedDataB[j + l * actualSize];
        }
        outputDistances[k] = distancesB[j];
        j++;
        k++;
    }
}

__global__ void printArr(float *data, int n, int dim, float *d_distances)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf("%f ", data[i + j * n]);
        }
        printf("Distance: %f\n", d_distances[i]);
    }
}

__global__ void mergeSort(float *data, int *labels, float *distances, long long n, int dim, long long threadSize,
                          long long sortedSize)
{
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
            iBlock = coRank(data, dim, distances, iSize, jSize, jOffset, iOffset, kBlock, n);
            iNextBlock = coRank(data, dim, distances, iSize, jSize, jOffset, iOffset, kNextBlock, n);

            jBlock = kBlock - iBlock + jOffset;
            jNextBlock = kNextBlock - iNextBlock + jOffset;
        }
    }

    __syncthreads();

    long long actualSize = 0;
    long long nBlock = 0, mBlock = 0;
    extern __shared__ float sharedArr[];
    float *distancesShared, *output, *outputDistances;
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
        distancesShared = sharedArr + 2 * actualSize * dim;
        outputDistances = distancesShared + actualSize;
        for (long long i = threadIdx.x; i < nBlock; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                sharedArr[i + j * actualSize] = data[(i + iBlock) + j * n];
            }
            distancesShared[i] = distances[i + iBlock];
        }
        for (long long i = threadIdx.x; i < mBlock; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                sharedArr[(i + nBlock) + j * actualSize] = data[(i + jBlock) + j * n];
            }
            distancesShared[i + nBlock] = distances[i + jBlock];
        }
    }

    __syncthreads();

    // shared memory for output
    output = sharedArr + actualSize * dim;

    long long k = threadIdx.x * threadSize;
    if (kBlock < n && k < actualSize)
    {
        // printf("actualSize: %lld, k: %lld nBlock: %lld, mBlock: %lld\n", actualSize, k, nBlock, mBlock);
        long long i = coRank(sharedArr, dim, distancesShared, nBlock, mBlock, nBlock, 0, k, actualSize);
        long long j = k - i + nBlock;
        long long kNext = min(k + threadSize, actualSize);
        long long iNext = coRank(sharedArr, dim, distancesShared, nBlock, mBlock, nBlock, 0, kNext, actualSize);
        long long jNext = kNext - iNext + nBlock;

        // sequential merge
        long long nn = iNext - i;
        long long mm = jNext - j;

        // printf("k: %lld, i: %lld, j: %lld, actualSize: %lld, mm: %lld, nn: %lld\n", k, i, j, actualSize, mm, nn);
        sequintialMergeSegment(sharedArr + i, sharedArr + j, output + k, distancesShared + i, distancesShared + j,
                               outputDistances + k, nn, mm, dim, actualSize);
    }
    __syncthreads();

    if (kBlock < n)
    {
        // copy output to data
        for (long long i = threadIdx.x; i < actualSize; i += blockDim.x)
        {
            for (int j = 0; j < dim; j++)
            {
                data[(i + kBlock) + j * n] = output[i + j * actualSize];
            }
            distances[i + kBlock] = outputDistances[i];
        }
    }
}
