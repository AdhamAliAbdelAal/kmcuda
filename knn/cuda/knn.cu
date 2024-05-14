#include "./knn.h"

int k = 0, n = 0, dim = 0;

void bubbleSortResult(double *output, double *target)
{
    for (int i = 0; i < k; i++)
    {
        for (int j = i + 1; j < k; j++)
        {
            double dist1 = 0;
            double dist2 = 0;
            for (int l = 0; l < dim; l++)
            {
                dist1 += (output[i * dim + l] - target[l]) * (output[i * dim + l] - target[l]);
                dist2 += (output[j * dim + l] - target[l]) * (output[j * dim + l] - target[l]);
            }
            if (dist1 > dist2)
            {
                for (int l = 0; l < dim; l++)
                {
                    double temp = output[i * dim + l];
                    output[i * dim + l] = output[j * dim + l];
                    output[j * dim + l] = temp;
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
            cout << output[i * dim + j] << " ";
        }

        cout << "Distance: ";
        double dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (output[i * dim + j] - target[j]) * (output[i * dim + j] - target[j]);
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

    double *data = NULL;
    int *labels = NULL;
    double *target = NULL;

    // Read data
    read_data(input_file, data, labels, target);

    int threadSize = 2 * k;

    // Print top 5 data
    // print_top(data, labels, n, target);

    // allocate device memory
    double *d_data, *d_data2, *d_target, *d_distances, *d_distances2;
    int *d_labels, *d_labels2;

    cudaMalloc(&d_data, sizeof(double) * n * dim);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_target, sizeof(double) * dim);
    cudaMalloc(&d_data2, sizeof(double) * n * dim);
    cudaMalloc(&d_labels2, sizeof(int) * n);
    cudaMalloc(&d_distances, sizeof(double) * n);
    cudaMalloc(&d_distances2, sizeof(double) * n);

    // copy data to device
    cudaMemcpy(d_data, data, sizeof(double) * n * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, sizeof(double) * dim, cudaMemcpyHostToDevice);

    // calculate distances
    long long calcDistThreadSize = 256;
    calcDistances<<<(n + calcDistThreadSize - 1) / calcDistThreadSize, calcDistThreadSize>>>(d_data, d_target,
                                                                                             d_distances, n, dim);

    // call kernel
    long long i = 0;
    while (n > k)
    {
        int threadsPerBlock = 256;
        long long blocksPerGrid = (n + threadSize - 1) / threadSize;

        if (i % 2 == 0)
        {
            knn<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_labels, d_distances, threadSize, n, dim, k, d_target,
                                                    d_data2, d_labels2, d_distances2);
        }
        else
        {
            knn<<<blocksPerGrid, threadsPerBlock>>>(d_data2, d_labels2, d_distances2, threadSize, n, dim, k, d_target,
                                                    d_data, d_labels, d_distances);
        }
        i++;
        long long numKs = n / (threadSize);
        long long rem = n % (threadSize);
        n = numKs * k;
        if (rem < k)
        {
            n += rem;
        }
        else
        {
            n += k;
        }
        cout << "n: " << n << endl;
        // wait for kernel to finish
        cudaDeviceSynchronize();
    }

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    // allocate memory for output
    double *output = (double *)malloc(sizeof(double) * k * dim);
    int *labelsOutput = (int *)malloc(sizeof(int) * k);

    // copy output from device to host
    if (i % 2 == 0)
    {
        cudaMemcpy(output, d_data, sizeof(double) * k * dim, cudaMemcpyDeviceToHost);
        cudaMemcpy(labelsOutput, d_labels, sizeof(int) * k, cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(output, d_data2, sizeof(double) * k * dim, cudaMemcpyDeviceToHost);
        cudaMemcpy(labelsOutput, d_labels2, sizeof(int) * k, cudaMemcpyDeviceToHost);
    }

    // sort output
    bubbleSortResult(output, target);

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