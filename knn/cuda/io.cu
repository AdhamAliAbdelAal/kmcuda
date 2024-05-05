#include "./knn.h"

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
      // set precision to 10 decimal places
      cout << fixed;
      cout.precision(10);
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