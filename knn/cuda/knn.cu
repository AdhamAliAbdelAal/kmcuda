#include "./knn.h"

long long k = 0, n = 0, dim = 0;

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
    int *indices = NULL;

    // Read data
    read_data(input_file, data, labels, target, indices);

    int threadSize = 2 * k;

    // Print top 5 data
    // print_top(data, labels, n, target);

    // allocate device memory
    double *d_data, *d_target, *d_distances, *d_distances2;
    int *d_indices1, *d_indices2;

    cudaMalloc(&d_data, sizeof(double) * n * dim);
    cudaMalloc(&d_target, sizeof(double) * dim);
    cudaMalloc(&d_distances, sizeof(double) * n);
    cudaMalloc(&d_distances2, sizeof(double) * n);
    cudaMalloc(&d_indices1, sizeof(int) * n);
    cudaMalloc(&d_indices2, sizeof(int) * n);

    // copy data to device
    cudaMemcpy(d_data, data, sizeof(double) * n * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, sizeof(double) * dim, cudaMemcpyHostToDevice);

    // create streams
    unsigned int num_streams = 8;
    cudaStream_t streams[8];
    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    int threadsPerBlock = 256;
    long long num_segments = num_streams;
    long long segment_size = (n + num_segments - 1) / num_segments;
    for (long long i = 0; i < num_segments; i++)
    {
        long long start = i * segment_size;
        long long end = min((i + 1) * segment_size, n);
        long long nsegment = end - start;
        if (nsegment <= 0)
            break;
        cudaMemcpyAsync(d_data + start * dim, data + start * dim, sizeof(double) * nsegment * dim,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_indices1 + start, indices + start, sizeof(int) * nsegment, cudaMemcpyHostToDevice,
                        streams[i]);
        long long calcDistThreadSize = 256;

        calcDistances<<<(nsegment + calcDistThreadSize - 1) / calcDistThreadSize, calcDistThreadSize, 0, streams[i]>>>(
            d_data + start * dim, d_target, d_distances + start, nsegment, dim);
    }
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(error) << endl;
        return 1;
    }
    // call kernel
    long long i = 0;
    while (n > k)
    {
        long long blocksPerGrid = (n + threadSize - 1) / threadSize;
        if (i % 2 == 0)
        {
            knn<<<blocksPerGrid, threadsPerBlock>>>(d_indices1, d_distances, threadSize, n, k, d_target, d_indices2,
                                                    d_distances2);
        }
        else
        {
            knn<<<blocksPerGrid, threadsPerBlock>>>(d_indices2, d_distances2, threadSize, n, k, d_target, d_indices1,
                                                    d_distances);
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
        // wait for kernel to finish
        cudaDeviceSynchronize();
    }

    // check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(error) << endl;
        return 1;
    }

    // allocate memory for output
    int *output = (int *)malloc(sizeof(int) * k);
    double *distances = (double *)malloc(sizeof(double) * k);
    // copy output from device to host
    if (i % 2 == 0)
    {
        cudaMemcpy(output, d_indices1, sizeof(int) * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(distances, d_distances, sizeof(double) * k, cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(output, d_indices2, sizeof(int) * k, cudaMemcpyDeviceToHost);
        cudaMemcpy(distances, d_distances2, sizeof(double) * k, cudaMemcpyDeviceToHost);
    }

    // sort output
    // bubbleSortResult(output, target);
    // print output
    for (int i = 0; i < k; i++)
    {
        cout << "Data " << i << ": ";
        for (int j = 0; j < dim; j++)
        {
            cout << fixed;
            cout.precision(10);
            cout << data[output[i] * dim + j] << " ";
        }

        cout << "Distance: ";
        double dist = 0;
        for (int j = 0; j < dim; j++)
        {
            dist += (data[output[i] * dim + j] - target[j]) * (data[output[i] * dim + j] - target[j]);
        }
        cout << fixed;
        cout.precision(10);
        cout << sqrt(dist) << " " << distances[i];
        cout << endl;
    }
    // free device memory
    cudaFree(d_data);
    cudaFree(d_target);
    cudaFree(d_distances);
    cudaFree(d_distances2);
    cudaFree(d_indices1);
    cudaFree(d_indices2);

    // Write output
    // write_data(output_file, output, labelsOutput);

    // Free memory
    free(data);
    free(labels);
    free(target);
    free(output);

    return 0;
}