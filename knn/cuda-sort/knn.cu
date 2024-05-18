#include "./knn.h"

long long n = 0;
int k = 0, dim = 0;

void bubbleSortResult(float *output, float *target)
{
    for (int i = 0; i < k; i++)
    {
        for (int j = i + 1; j < k; j++)
        {
            float dist1 = 0;
            float dist2 = 0;
            for (int l = 0; l < dim; l++)
            {
                dist1 += (output[i + l * n] - target[l]) * (output[i + l * n] - target[l]);
                dist2 += (output[j + l * n] - target[l]) * (output[j + l * n] - target[l]);
            }
            if (dist1 > dist2)
            {
                for (int l = 0; l < dim; l++)
                {
                    float temp = output[i + l * n];
                    output[i + l * n] = output[j + l * n];
                    output[j + l * n] = temp;
                }
            }
        }
    }

    // print sorted output
    cout << "Sorted output:" << endl;
    for (int i = 0; i < k; i++)
    {
        cout << "Data " << i << ": ";
        for (int j = 0; j < dim; j++)
        {
            cout << fixed;
            cout.precision(10);
            cout << output[i + j * n] << " ";
        }

        cout << "Distance: ";
        float dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (output[i + j * n] - target[j]) * (output[i + j * n] - target[j]);
        }
        cout << fixed;
        cout.precision(10);
        cout << sqrt(dist);
        cout << endl;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "Usage: ./knn input_file output_file" << endl;
        return 1;
    }
    string input_file = argv[1];
    string output_file = argv[2];

    float *data = NULL;
    int *labels = NULL;
    float *target = NULL;

    // Read data
    read_data(input_file, data, labels, target);

    // Print top 5 data
    // print_top(data, labels, n, target);

    // allocate device memory
    float *d_data, *d_target, *d_distances;
    int *d_labels;

    cudaMalloc(&d_data, sizeof(float) * n * dim);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_target, sizeof(float) * dim);
    cudaMalloc(&d_distances, sizeof(float) * n);

    // copy data to device
    cudaMemcpy(d_target, target, sizeof(float) * dim, cudaMemcpyHostToDevice);

    long long sortedSize = 8;
    long long bNumThreads = 64;
    long long bNumBlocks = 1;
    long long segementSize = bNumBlocks * bNumThreads * sortedSize;
    long long numSegements = (n + segementSize - 1) / segementSize;
    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * numSegements);
    for (int i = 0; i < numSegements; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    for (int i = 0; i < numSegements; i++)
    {
        long long start = i * segementSize;
        long long end = min((i + 1) * segementSize, n);
        long long size = end - start;
        // copy data to device
        for (int j = 0; j < dim; j++)
        {
            cudaMemcpyAsync(d_data + start + j * n, data + start + j * n, sizeof(float) * size, cudaMemcpyHostToDevice,
                            streams[i]);
        }

        // calculate the distances
        calcDistances<<<(size + 255) / 256, 256, 0, streams[i]>>>(d_data + start, d_target, d_distances + start, size,
                                                                  dim, n);
        // bubble sort
        bubbleSort<<<bNumBlocks, bNumThreads, 0, streams[i]>>>(d_data + start, d_labels + start, d_distances + start,
                                                               size, dim, sortedSize, n);
    }
    // sync
    cudaDeviceSynchronize();

    // // copy data back to host
    // cudaMemcpy(data, d_data, sizeof(float) * n * dim, cudaMemcpyDeviceToHost);

    // //  print data and distance
    // for (int i = 0; i < 10; i++)
    // {
    //     for (int j = 0; j < dim; j++)
    //     {
    //         printf("%f ", data[i + j * n]);
    //     }
    //     // print distance
    //     float dist = 0;
    //     for (int j = 0; j < dim; j++)
    //     {
    //         dist += (data[i + j * n] - target[j]) * (data[i + j * n] - target[j]);
    //     }
    //     printf("Distance: %f\n", sqrt(dist));
    // }
    // printf("++++++++++++++++++++++++++++++\n");
    long long elementsPerThread = 4;
    long long numThreads = 256;
    long long elementsPerBlock = numThreads * elementsPerThread;
    long long maxElementsPerBlock = min(numThreads * elementsPerThread, sortedSize * 2);
    long long numBlocksPerSortedSize = (2 * sortedSize + maxElementsPerBlock - 1) / maxElementsPerBlock;
    long long numSortedSize = (n + 2 * sortedSize - 1) / (2 * sortedSize);
    long long numBlocks = numSortedSize * numBlocksPerSortedSize;
    long long sharedMemSize = 2 * elementsPerBlock * dim * sizeof(float) + 2 * elementsPerBlock * sizeof(float);
    while (sortedSize < n)
    {
        // maxElementsPerBlock = min(numThreads * elementsPerThread, sortedSize * 2);
        // numBlocksPerSortedSize = (2 * sortedSize + maxElementsPerBlock - 1) / maxElementsPerBlock;
        // numSortedSize = (n + 2 * sortedSize - 1) / (2 * sortedSize);
        // numBlocks = numSortedSize * numBlocksPerSortedSize;
        // printf("== numBlocks: %lld\n", numBlocks);
        // printf("== 2 * elementsPerBlock: %lld\n", 2 * elementsPerBlock);
        mergeSort<<<numBlocks, numThreads, sharedMemSize>>>(d_data, d_labels, d_distances, n, dim, elementsPerThread,
                                                            sortedSize);
        cudaError_t cudaStatus = cudaDeviceSynchronize();

        // copy data back to host
        cudaMemcpy(data, d_data, sizeof(float) * n * dim, cudaMemcpyDeviceToHost);

        if (cudaStatus != cudaSuccess)
        {
            printf("Error: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }
        sortedSize *= 2;
        // printf("== sortedSize: %lld\n", sortedSize);
        // printArr<<<1, 1>>>(d_data, 5, dim, d_distances);
        // break;
    }

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    // allocate memory for output
    // float *output = (float *)malloc(sizeof(float) * n * dim);
    int *labelsOutput = (int *)malloc(sizeof(int) * k);
    float *outDistances = (float *)malloc(sizeof(float) * n);
    // copy output from device to host
    cudaMemcpy(data, d_data, sizeof(float) * n * dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(outDistances, d_distances, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // print top k data
    for (int i = 0; i < k; i++)
    {
        cout << "Data " << i << ": ";
        for (int j = 0; j < dim; j++)
        {
            cout << fixed;
            cout.precision(10);
            cout << data[i + j * n] << " ";
        }

        cout << "Distance: ";
        float dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (data[i + j * n] - target[j]) * (data[i + j * n] - target[j]);
        }
        cout << fixed;
        cout.precision(10);
        cout << sqrt(dist) << " " << outDistances[i];
        cout << endl;
    }

    // free device memory
    cudaFree(d_data);
    cudaFree(d_labels);
    cudaFree(d_target);
    cudaFree(d_distances);

    // Write output
    write_data(output_file, data, labelsOutput);

    // Free memory
    free(data);
    free(labels);
    free(target);
    free(labelsOutput);

    return 0;
}