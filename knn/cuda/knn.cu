#include <iostream>
using namespace std;

int k, n, dim;

bool read_data(string input_file, double**& data, int*& labels) {
  FILE* file = freopen(input_file.c_str(), "r", stdin);
  if (file == NULL) {
    cout << "Cannot open file " << input_file << endl;
    return false;
  }
  cout << "Reading data from " << input_file << endl;
  cin >> k >> n >> dim;
  cout << "k = " << k << ", n = " << n << ", dim = " << dim << endl;
  data = (double**)malloc(sizeof(double*) * n);
  labels = (int*)malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) {
    data[i] = (double*)malloc(sizeof(double) * dim);
    for (int j = 0; j < dim; j++) {
      cin >> data[i][j];
    }
    cin >> labels[i];
  }
  fclose(file);
  return true;
}

void print_top(double** data, int* labels, int n) {
  for (int i = 0; i < n; i++) {
    cout << "Data " << i << ": ";
    for (int j = 0; j < dim; j++) {
      cout << data[i][j] << " ";
    }
    cout << "Label: " << labels[i] << endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "Usage: ./knn input_file output_file" << endl;
    return 1;
  }
  string input_file = argv[1];
  string output_file = argv[2];

  double** data = NULL;
  int* labels = NULL;

  // Read data
  read_data(input_file, data, labels);

  // Print top 5 data
  print_top(data, labels, 5);
  return 0;
}