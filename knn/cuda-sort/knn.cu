#include "./knn.h"

int k = 0, n = 0, dim = 0;

void bubbleSortResult(float* output, float* target)
{
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            float dist1 = 0;
            float dist2 = 0;
            for (int l = 0; l < dim; l++) {
                dist1 += (output[i * dim + l] - target[l]) * (output[i * dim + l] - target[l]);
                dist2 += (output[j * dim + l] - target[l]) * (output[j * dim + l] - target[l]);
            }
            if (dist1 > dist2) {
                for (int l = 0; l < dim; l++) {
                    float temp = output[i * dim + l];
                    output[i * dim + l] = output[j * dim + l];
                    output[j * dim + l] = temp;
                }
            }
        }
    }

    // print sorted output
    cout << "Sorted output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << "Data " << i << ": ";
        for (int j = 0; j < dim; j++) {
            cout << fixed;
            cout.precision(10);
            cout << output[i * dim + j] << " ";
        }

        cout << "Distance: ";
        float dist = 0;
        for (int j = 0; j < dim; j++) {
            dist += (output[i * dim + j] - target[j]) * (output[i * dim + j] - target[j]);
        }
        cout << fixed;
        cout.precision(10);
        cout << sqrt(dist);
        cout << endl;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "Usage: ./knn input_file output_file" << endl;
        return 1;
    }
    string input_file = argv[1];
    string output_file = argv[2];

    float* data = NULL;
    int* labels = NULL;
    float* target = NULL;

    // Read data
    read_data(input_file, data, labels, target);

    // Print top 5 data
    // print_top(data, labels, n, target);

    // allocate device memory
    float *d_data, *d_target;
    int* d_labels;

    cudaMalloc(&d_data, sizeof(float) * n * dim);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_target, sizeof(float) * dim);

    // copy data to device
    cudaMemcpy(d_data, data, sizeof(float) * n * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, sizeof(float) * dim, cudaMemcpyHostToDevice);
    long long sortedSize = 30;
    long long bNumThreads = 32;
    long long bNumBlocks = (n + sortedSize - 1) / (sortedSize);
    // call merge sort kernel
    bubbleSort<<<bNumBlocks, bNumThreads>>>(d_data, d_labels, n, dim, d_target, sortedSize);

    // sync
    cudaDeviceSynchronize();

    // copy data back to host
    cudaMemcpy(data, d_data, sizeof(float) * n * dim, cudaMemcpyDeviceToHost);

    //  print data and distance
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < dim; j++) {
    //         printf("%f ", data[i * dim + j]);
    //     }
    //     // print distance
    //     float dist = 0;
    //     for (int j = 0; j < dim; j++) {
    //         dist += (data[i * dim + j] - target[j]) * (data[i * dim + j] - target[j]);
    //     }
    //     printf("Distance: %f\n", sqrt(dist));
    // }
    // printf("++++++++++++++++++++++++++++++\n");
    long long elementsPerThread = 50;
    long long numThreads = 32;
    long long numBlocks = (n + sortedSize * 2 - 1) / (sortedSize * 2);
    long long elementsPerBlock = numThreads * elementsPerThread;
    while (sortedSize < n) {
        mergeSort<<<numBlocks, numThreads, 2 * elementsPerBlock * dim * sizeof(float) + 2 * sizeof(long long)>>>(d_data, d_labels, d_target, n, dim, elementsPerThread, sortedSize);
        cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(cudaStatus));
        }
        sortedSize *= 2;
        printf("== sortedSize: %lld\n", sortedSize);
        // printArr<<<1, 1>>>(d_data, 5, dim, d_target);
        // break;
    }

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "Error: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    // allocate memory for output
    float* output = (float*)malloc(sizeof(float) * k * dim);
    int* labelsOutput = (int*)malloc(sizeof(int) * k);

    // copy output from device to host
    cudaMemcpy(output, d_data, sizeof(float) * k * dim, cudaMemcpyDeviceToHost);

    // print top k data
    for (int i = 0; i < k; i++) {
        cout << "Data " << i << ": ";
        for (int j = 0; j < dim; j++) {
            cout << fixed;
            cout.precision(10);
            cout << output[i * dim + j] << " ";
        }

        cout << "Distance: ";
        float dist = 0;
        for (int j = 0; j < dim; j++) {
            dist += (output[i * dim + j] - target[j]) * (output[i * dim + j] - target[j]);
        }
        cout << fixed;
        cout.precision(10);
        cout << sqrt(dist);
        cout << endl;
    }

    // free device memory
    cudaFree(d_data);
    cudaFree(d_labels);

    // Write output
    write_data(output_file, output, labelsOutput);

    // Free memory
    free(data);
    free(labels);
    free(target);
    free(output);
    free(labelsOutput);

    return 0;
}