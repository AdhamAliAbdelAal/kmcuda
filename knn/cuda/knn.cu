#include <iostream>
using namespace std;

int k, n, dim;

bool read_data(string input_file, double*& data, int*& labels) {
  FILE* file = freopen(input_file.c_str(), "r", stdin);
  if (file == NULL) {
    cout << "Cannot open file " << input_file << endl;
    return false;
  }
  cout << "Reading data from " << input_file << endl;
  cin >> k >> n >> dim;
  cout << "k = " << k << ", n = " << n << ", dim = " << dim << endl;
  data = (double*)malloc(sizeof(double) * n * dim);
  labels = (int*)malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      cin >> data[i * dim + j];
    }
    cin >> labels[i];
  }
  fclose(file);
  return true;
}

bool write_data(string output_file, double* output) {
  FILE* file = freopen(output_file.c_str(), "w", stdout);
  if (file == NULL) {
    cout << "Cannot open file " << output_file << endl;
    return false;
  }
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < dim; j++) {
      cout << output[i * dim + j] << " ";
    }
    cout << endl;
  }
  fclose(file);
  return true;
}

void print_top(double* data, int* labels, int n) {
  cout << "Top " << n << " data:" << endl;
  for (int i = 0; i < n; i++) {
    cout << "Data " << i << ": ";
    for (int j = 0; j < dim; j++) {
      cout << data[i * dim + j] << " ";
    }
    cout << "Label: " << labels[i] << endl;
  }
}

__global__ void knn(double* data, int* labels, int n, int dim, double* output,
                    int k) {
  // print thread info
  printf("Thread %d %d\n", threadIdx.x, blockIdx.x);

  // print top 5 data
  for (int i = 0; i < 5; i++) {
    printf("Data %d: ", i);
    for (int j = 0; j < dim; j++) {
      printf("%f ", data[i * dim + j]);
    }
    printf("Label: %d\n", labels[i]);
  }
  printf("Finished printing top 5 data\n");
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Usage: ./knn input_file output_file" << endl;
    return 1;
  }
  string input_file = argv[1];
  string output_file = argv[2];

  double* data = NULL;
  int* labels = NULL;

  // Read data
  read_data(input_file, data, labels);

  // Print top 5 data
  print_top(data, labels, 2);

  // allocate memory for output
  double* output = (double*)malloc(sizeof(double) * k * dim);

  // add dummy data to output
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < dim; j++) {
      output[i * dim + j] = 0;
    }
  }

  // allocate device memory
  double* d_data;
  int* d_labels;
  double* d_output;
  cudaMalloc(&d_data, sizeof(double) * n * dim);
  cudaMalloc(&d_labels, sizeof(int) * n);
  cudaMalloc(&d_output, sizeof(double) * k * dim);

  // copy data to device
  cudaMemcpy(d_data, data, sizeof(double) * n * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);

  // call kernel
  knn<<<1, 1>>>(d_data, d_labels, n, dim, d_output, k);

  // wait for kernel to finish
  cudaDeviceSynchronize();

  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    cout << "Error: " << cudaGetErrorString(error) << endl;
    return 1;
  }

  // copy output back to host
  cudaMemcpy(output, d_output, sizeof(double*) * k, cudaMemcpyDeviceToHost);

  // free device memory
  cudaFree(d_data);
  cudaFree(d_labels);
  cudaFree(d_output);

  // Write output
  write_data(output_file, output);

  // Free memory
  free(data);

  return 0;
}