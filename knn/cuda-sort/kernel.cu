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

__global__ void bubbleSort(int *indices, float *distances, long long n, int sizeToSort, long long offset)
{

    long long start = (blockIdx.x * blockDim.x + threadIdx.x) * sizeToSort;
    long long end = start + sizeToSort < n ? start + sizeToSort : n;
    if (start >= n)
    {
        return;
    }
    // init indices
    for (int i = start; i < end; i++)
    {
        indices[i] = i + offset;
    }
    // bubble sort (swich distances and indices)
    for (int i = start; i < end; i++)
    {
        for (int j = start; j < end - i + start - 1; j++)
        {
            if (distances[j] > distances[j + 1])
            {
                float temp = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = temp;
                int tempIdx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tempIdx;
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

__device__ long long coRank(float *distances, long long n, long long m, long long jOffset, long long iOffset,
                            long long k)
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

__device__ void sequintialMergeSegment(int *sharedIndicesA, int *sharedIndicesB, int *outputIndices, float *distancesA,
                                       float *distancesB, float *outputDistances, long long n, long long m)
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
            outputIndices[k] = sharedIndicesA[i];
            outputDistances[k] = dist1;
            i++;
        }
        else
        {
            outputIndices[k] = sharedIndicesB[j];
            outputDistances[k] = dist2;
            j++;
        }
        k++;
    }

    while (i < n)
    {
        outputIndices[k] = sharedIndicesA[i];
        outputDistances[k] = distancesA[i];
        i++;
        k++;
    }

    while (j < m)
    {
        outputIndices[k] = sharedIndicesB[j];
        outputDistances[k] = distancesB[j];
        j++;
        k++;
    }
}

__global__ void printArr(float *data, int k, int dim, float *d_distances, int *indices, long long n, float *target)
{
    // print top k data
    for (int i = 0; i < k; i++)
    {
        printf("Data %d: ", i);
        for (int j = 0; j < dim; j++)
        {
            printf("%f ", data[indices[i] + j * n]);
        }

        printf("Distance: ");
        float dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (data[indices[i] + j * n] - target[j]) * (data[indices[i] + j * n] - target[j]);
        }
        printf("%f %f\n", sqrt(dist), d_distances[i]);
    }
    // print last k data
    for (int i = n - k; i < n; i++)
    {
        printf("Data %d: ", i);
        for (int j = 0; j < dim; j++)
        {
            printf("%f ", data[indices[i] + j * n]);
        }

        printf("Distance: ");
        float dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (data[indices[i] + j * n] - target[j]) * (data[indices[i] + j * n] - target[j]);
        }
        printf("%f %f\n", sqrt(dist), d_distances[i]);
    }
    printf("\n");
}

__global__ void mergeSort(int *indices, float *distances, long long n, long long threadSize, long long sortedSize)
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
            iBlock = coRank(distances, iSize, jSize, jOffset, iOffset, kBlock);
            iNextBlock = coRank(distances, iSize, jSize, jOffset, iOffset, kNextBlock);

            jBlock = kBlock - iBlock + jOffset;
            jNextBlock = kNextBlock - iNextBlock + jOffset;
        }
    }

    __syncthreads();

    long long actualSize = 0;
    long long nBlock = 0, mBlock = 0;
    extern __shared__ int sharedArr[];
    int *output;
    float *distancesShared, *outputDistances;
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
        output = sharedArr + actualSize;
        distancesShared = (float *)(sharedArr + 2 * actualSize);
        outputDistances = distancesShared + actualSize;
        for (long long i = threadIdx.x; i < nBlock; i += blockDim.x)
        {
            sharedArr[i] = indices[i + iBlock];
            distancesShared[i] = distances[i + iBlock];
        }
        for (long long i = threadIdx.x; i < mBlock; i += blockDim.x)
        {
            sharedArr[i + nBlock] = indices[i + jBlock];
            distancesShared[i + nBlock] = distances[i + jBlock];
        }
    }

    __syncthreads();

    long long k = threadIdx.x * threadSize;
    if (kBlock < n && k < actualSize)
    {
        // printf("actualSize: %lld, k: %lld nBlock: %lld, mBlock: %lld\n", actualSize, k, nBlock, mBlock);
        long long i = coRank(distancesShared, nBlock, mBlock, nBlock, 0, k);
        long long j = k - i + nBlock;
        long long kNext = min(k + threadSize, actualSize);
        long long iNext = coRank(distancesShared, nBlock, mBlock, nBlock, 0, kNext);
        long long jNext = kNext - iNext + nBlock;

        // sequential merge
        long long nn = iNext - i;
        long long mm = jNext - j;

        // printf("k: %lld, i: %lld, j: %lld, actualSize: %lld, mm: %lld, nn: %lld\n", k, i, j, actualSize, mm, nn);
        sequintialMergeSegment(sharedArr + i, sharedArr + j, output + k, distancesShared + i, distancesShared + j,
                               outputDistances + k, nn, mm);
    }
    __syncthreads();

    if (kBlock < n)
    {
        // copy output to data
        for (long long i = threadIdx.x; i < actualSize; i += blockDim.x)
        {
            indices[i + kBlock] = output[i];
            distances[i + kBlock] = outputDistances[i];
        }
    }
}
