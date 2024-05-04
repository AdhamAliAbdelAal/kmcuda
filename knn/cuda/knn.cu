#include <iostream>
using namespace std;

int k, n, dim;

bool read_data(string input_file, double *&data, int *&labels,
               double *&target) {
  FILE *file = freopen(input_file.c_str(), "r", stdin);
  if (file == NULL) {
    cout << "Cannot open file " << input_file << endl;
    return false;
  }
  cout << "Reading data from " << input_file << endl;
  cin >> k >> n >> dim;
  cout << "k = " << k << ", n = " << n << ", dim = " << dim << endl;
  data = (double *)malloc(sizeof(double) * n * dim);
  labels = (int *)malloc(sizeof(int) * n);
  target = (double *)malloc(sizeof(double) * dim);

  for (int i = 0; i < dim; i++) {
    cin >> target[i];
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) {
      cin >> data[i * dim + j];
    }
    cin >> labels[i];
  }
  fclose(file);
  return true;
}

bool write_data(string output_file, double *output) {
  FILE *file = freopen(output_file.c_str(), "w", stdout);
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

void print_top(double *data, int *labels, int n, double *target) {
  cout << "Target data: ";
  for (int i = 0; i < dim; i++) {
    cout << target[i] << " ";
  }
  cout << endl;
  cout << "Top " << n << " data:" << endl;
  for (int i = 0; i < n; i++) {
    cout << "Data " << i << ": ";
    for (int j = 0; j < dim; j++) {
      cout << data[i * dim + j] << " ";
    }
    cout << "Label: " << labels[i] << endl;
  }
}

__device__ double euclidean_distance(double *data, int idx, double *target,
                                     int dim) {
  double sum = 0;
  printf("element %d: (", idx);
  for (int i = 0; i < dim; i++) {
    // print element
    printf("%f, ", data[idx * dim + i]);
    sum +=
        (data[idx * dim + i] - target[i]) * (data[idx * dim + i] - target[i]);
  }
  printf(", sum = %f)\n", sqrt(sum));
  return sqrt(sum);
}

__device__ void sortElements(double *data, int startElement, int endElement,
                             double *target, int k, int *nearesrtNeighborsIdxs,
                             int dim) {
  double *distances = (double *)malloc(sizeof(double) * k);
  for (int i = startElement; i <= endElement; i++) {
    double dist = euclidean_distance(data, i, target, dim);
    if (i < k) {
      distances[i] = dist;
      nearesrtNeighborsIdxs[i] = i;
    } else {
      int maxIdx = 0;
      for (int j = 1; j < k; j++) {
        if (distances[j] > distances[maxIdx]) {
          maxIdx = j;
        }
      }
      if (dist < distances[maxIdx]) {
        distances[maxIdx] = dist;
        nearesrtNeighborsIdxs[maxIdx] = i;
      }
    }
  }
}

__global__ void knn(double *data, int *labels, int threadSize, int n, int dim,
                    int k, double *target) {
  // print thread info
  printf("Thread %d %d\n", threadIdx.x, blockIdx.x);

  int startElement = (blockIdx.x * blockDim.x + threadIdx.x) * threadSize;
  int endElement =
      startElement + threadSize < n ? startElement + threadSize : n - 1;
  // pritn start and end element
  printf("Start element: %d, End element: %d\n", startElement, endElement);
  int *nearesrtNeighborsIdxs = (int *)malloc(sizeof(int) * k);
  sortElements(data, startElement, endElement, target, k, nearesrtNeighborsIdxs,
               dim);
  // DEBUG print nearest neighbors
  for (int i = 0; i < k; i++) {
    printf("Nearest neighbor %d: %d\n", i, nearesrtNeighborsIdxs[i]);
  }

  // copy elements to data again
  int startElementCopy = (blockIdx.x * blockDim.x + threadIdx.x) * k;
  int endElementCopy = startElementCopy + k < n ? startElementCopy + k : n - 1;
  for (int i = startElementCopy; i <= endElementCopy; i++) {
    for (int j = 0; j < dim; j++) {
      data[i * dim + j] = data[nearesrtNeighborsIdxs[i] * dim + j];
    }
  }
  free(nearesrtNeighborsIdxs);
}

int main(int argc, char **argv) {
  if (argc != 3) {
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

  // Print top 5 data
  // print_top(data, labels, n, target);

  // allocate device memory
  double *d_data;
  int *d_labels;
  double *d_target;
  cudaMalloc(&d_data, sizeof(double) * n * dim);
  cudaMalloc(&d_labels, sizeof(int) * n);
  cudaMalloc(&d_target, sizeof(double) * dim);

  // copy data to device
  cudaMemcpy(d_data, data, sizeof(double) * n * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_target, target, sizeof(double) * dim, cudaMemcpyHostToDevice);

  // call kernel
  // int threadSize = 2 * k;
  int threadSize = n;
  knn<<<1, 1>>>(d_data, d_labels, threadSize, n, dim, k, d_target);

  // wait for kernel to finish
  cudaDeviceSynchronize();

  // check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    cout << "Error: " << cudaGetErrorString(error) << endl;
    return 1;
  }

  // allocate memory for output
  double *output = (double *)malloc(sizeof(double) * k * dim);

  // copy output from device to host
  cudaMemcpy(output, d_data, sizeof(double) * k * dim, cudaMemcpyDeviceToHost);

  // free device memory
  cudaFree(d_data);
  cudaFree(d_labels);

  // Write output
  write_data(output_file, output);

  // Free memory
  free(data);

  return 0;
}