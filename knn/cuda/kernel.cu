
__device__ double euclidean_distance(double* data, int idx, double* target,
    int dim)
{
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += (data[idx * dim + i] - target[i]) * (data[idx * dim + i] - target[i]);
    }
    return sqrt(sum);
}

__device__ void sortElements(double* data, int startElement, int endElement,
    double* target, int k, int* nearesrtNeighborsIdxs,
    int dim)
{
    double* distances = (double*)malloc(sizeof(double) * k);
    for (int i = startElement; i <= endElement; i++) {
        double dist = euclidean_distance(data, i, target, dim);
        if (i - startElement < k) {
            distances[i - startElement] = dist;
            nearesrtNeighborsIdxs[i - startElement] = i;
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

__global__ void knn(double* data, int* labels, int threadSize, int n, int dim,
    int k, double* target, double* output, int* labelsOutput)
{
    // print thread info
    int startElement = (blockIdx.x * blockDim.x + threadIdx.x) * threadSize;
    int endElement = startElement + threadSize < n ? startElement + threadSize - 1 : n - 1;
    int totalElements = endElement - startElement + 1;
    k = k < totalElements ? k : totalElements;

    if (startElement < n) {
        int* nearesrtNeighborsIdxs = (int*)malloc(sizeof(int) * k);
        sortElements(data, startElement, endElement, target, k,
            nearesrtNeighborsIdxs, dim);

        // copy elements to data again
        int startElementCopy = (blockIdx.x * blockDim.x + threadIdx.x) * k;
        int endElementCopy = startElementCopy + k - 1;
        for (int i = startElementCopy; i <= endElementCopy; i++) {
            for (int j = 0; j < dim; j++) {
                output[i * dim + j] = data[nearesrtNeighborsIdxs[i - startElementCopy] * dim + j];
            }
            labelsOutput[i] = labels[nearesrtNeighborsIdxs[i - startElementCopy]];
        }
        free(nearesrtNeighborsIdxs);
    }
}