#include <stdio.h> //provides input / output functions (like printf)
#include <stdlib.h> //provides general utility functions (like malloc)
#include <stdbool.h> //provides boolean data types and values
#include <string.h> //provides string functions (like memcpy)
#include <math.h> //provides mathematical functions (like pow)
#include <iostream> //standard C++ input/output stream (cout, etc)
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#define NUM_ROWS 100000
#define LINE_SIZE 100
#define NUM_ITERATIONS 100
#define MAX_INPUT_VALUE 100
#define NUM_CLUSTERS 3
#define NUM_SEEDS 32
#define EPSILON_PERCENT 0.02
#define TEST_ITERATIONS 2


#define LOG if (false)

#define LABELS_LINE(cudaPitchedPtr, row) ((int*) (((char*)((cudaPitchedPtr).ptr)) + (row) * (cudaPitchedPtr).pitch))
#define CENTROIDS_LINE(cudaPitchedPtr, row) ((float*) (((char*)((cudaPitchedPtr).ptr)) + (row) * (cudaPitchedPtr).pitch))

struct core_params_t {
    float* line; // IN
    int line_size; // IN
    int* labels; // IN/OUT
    float inertia; // IN/OUT
    float* centroids; // IN/OUT
    int num_clusters; // IN
    int seed; // IN
};

#define DECLARE_CORE_PARAMS(core_params) float* line = (core_params).line; \
                                        int line_size = (core_params).line_size; \
                                        int* labels = (core_params).labels; \
                                        float inertia = (core_params).inertia; \
                                        float* centroids = (core_params).centroids; \
                                        int num_clusters = (core_params).num_clusters; \
                                        int seed = (core_params).seed;

// find a minimum value
__device__ float find_min(const float* arr, int size, int* outIndex = NULL) {
    float res = -1;
    int min_index = 0;
    for (int i = 0; i < size; i++) {
        if (res == -1 || res > arr[i]) {
            res = arr[i];
            if (outIndex != NULL)
                *outIndex = i;
        }
    }
    return res;
}

// find a maxiumum value
__device__ float find_max(const float* arr, int size) {
    float res = -1;
    for (int i = 0; i < size; i++) {
        if (res == -1 || res < arr[i])
            res = arr[i];
    }
    return res;
}

__device__ int map_to_index(int val, int* val2indexMap, int& num_mapped, int* index2countMap) {

    for (int i = 0; i < num_mapped; i++) { //loop entries
        // found => update count & return index
        if (val == val2indexMap[i]) {
            if (index2countMap != NULL)
                index2countMap[i]++; //update count (if provided)
            return i; //return index
        }
    }
    // not found => add & return
    val2indexMap[num_mapped] = val; //add value at end
    if (index2countMap != NULL)
        index2countMap[num_mapped] = 1; //add to count (if provided)
    return num_mapped++;
}


__device__ void assignLabels(core_params_t& core_params) {
    DECLARE_CORE_PARAMS(core_params); // extracts members of core_params into individual variables
    for (int i = 0; i < line_size; i++) { //loop over each data point
        float elt = line[i];
        int bestCentroid = 0; // initial value
        //calculate distance between the datapoint and centroid
        float bestDist = round(abs(elt - centroids[bestCentroid]));
        for (int j = 0; j < num_clusters; j++) {
            float currDist = round(abs(elt - centroids[j]));
            //update best centroid and best distance
            if (currDist < bestDist) {
                bestCentroid = j;
                bestDist = currDist;
            }
        }
        labels[i] = bestCentroid; //assign label to best / closest centroid
    }
}

__device__ void sort_centroids(float* centroids, int num_clusters) {
    for (int i = 1; i < num_clusters; ++i) {
        float key = centroids[i];
        int j = i - 1;

        // Move elements that are greater than key to one position ahead
        while (j >= 0 && centroids[j] > key) {
            centroids[j + 1] = centroids[j];
            j = j - 1;
        }
        centroids[j + 1] = key;
    }
}


__device__ void updateCentroids(core_params_t& core_params) {
    DECLARE_CORE_PARAMS(core_params);

    for (int i = 0; i < NUM_CLUSTERS; i++) {
        int count = 0;
        float sum = 0;
        float centroid = centroids[i];
        for (int j = 0; j < LINE_SIZE; j++) {
            int label = labels[j];
            if (i == label) {
                sum += line[j];
                count += 1;
            }
        }
        if (count != 0) {
            centroids[i] = sum / count;
        }
    }
}

__device__ float calcInertia(core_params_t& core_params) {
    DECLARE_CORE_PARAMS(core_params);
    float sse = 0.0;
    for (int i = 0; i < num_clusters; i++) {
        float centroid = centroids[i];
        for (int j = 0; j < line_size; j++) {
            if (labels[j] == i) {
                float diff = line[j] - centroid;
                sse += diff * diff;
            }
        }
    }
    return sse;
    //     printf("Inertia: %0.2f \n", inertia);
}

__device__ float findYardStick(core_params_t& core_params) {
    DECLARE_CORE_PARAMS(core_params)
    float min = MAX_INPUT_VALUE;
    float yardStick = 0;
    for (int i = 1; i < NUM_CLUSTERS; i++) {
        float distance = centroids[i] - centroids[i - 1]; // assumes centroids is sorted
        if (distance < min) {
            min = distance;
        }
    }
    yardStick = min;
    return yardStick;
}

__device__ bool converged(core_params_t& core_params, const float* oldCentroids, const float epsilon) {
    DECLARE_CORE_PARAMS(core_params);
    // printf("Epsilon %0.2f \n", epsilon);
    for (int i = 0; i < NUM_CLUSTERS; i++) { // assuming centroids is sorted
        if (epsilon < abs(centroids[i] - oldCentroids[i])) {
            // printf("Current difference between new and old centroids %0.2f, \n", abs(centroids[i] - oldCentroids[i]));
            return false;
        }
    }
    return true;
}


__device__ bool allSame(const float* arr1, const float* arr2, int size) { //TODO: Improve convergence condition
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}


__device__ void kmeans(core_params_t& core_params) {
    DECLARE_CORE_PARAMS(core_params);

    sort_centroids(centroids, num_clusters);
    float yard_stick = findYardStick(core_params);
    // printf("Yard stick %0.2f \n", yard_stick);
    float oldCentroids[NUM_CLUSTERS]{};

    // epsilon application is simple

    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        bool stop = (iteration > 0 && (converged(core_params, oldCentroids, yard_stick * EPSILON_PERCENT)));
        if (stop)
            break;

        memcpy(oldCentroids, core_params.centroids, NUM_CLUSTERS * sizeof(float));

        assignLabels(core_params);
        updateCentroids(core_params);
    }

    free(oldCentroids);
}

__device__ void largest_smallest_distance(core_params_t& core_params, int iter, float centroids_row[], float data[], int data_length) {
    // For each next chosen centroid, measure distance to previously chosen centroids
    // and choose next centroid by the largest distance of the min of distances to other chosen centroids.
    DECLARE_CORE_PARAMS(core_params);

    float min = 0;
    int next_centroid_index = -1;

    for (int i = 0; i < data_length; i++) {
        float min_distance = MAX_INPUT_VALUE;

        for (int j = 0; j < iter; j++) {
            float distance = abs(data[i] - centroids_row[j]);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        if (min_distance > min) {
            min = min_distance;
            next_centroid_index = i;
        }
    }
    centroids_row[iter] = data[next_centroid_index];
    float temp = data[data_length - 1];
    data[data_length - 1] = data[next_centroid_index];
    data[next_centroid_index] = temp;
}

__device__ void getInitialCentroids(core_params_t& core_params, float seed_centroids[NUM_SEEDS][NUM_CLUSTERS]) {
    DECLARE_CORE_PARAMS(core_params);

    unsigned long long device_seed = clock64();
    curandState state;
    curand_init(device_seed, seed, seed, &state);

    float data[LINE_SIZE]{};
    int data_length = line_size;

    for (int i = 0; i < LINE_SIZE; i++) {
        data[i] = line[i];
    }

    for (int i = 0; i < NUM_SEEDS; i++) {
        int random_index = (int)((curand_uniform_double(&state)) * data_length);
        seed_centroids[i][0] = data[random_index];
        float temp = data[data_length - 1];
        data[data_length - 1] = data[random_index];
        data[random_index] = temp;
        data_length -= 1;

        for (int j = 1; j <= num_clusters - 1; j++) {
            largest_smallest_distance(core_params, j, seed_centroids[i], data, data_length);
            data_length -= 1;
        }
    }
}

__global__ void cuda_kmeans(float* input, cudaPitchedPtr outputLabels, cudaPitchedPtr outputCentroids) {
    int row = blockIdx.x;
    int seed = threadIdx.x;

    // Pre-allocate arrays in device memory - once per block!
    __shared__ float input_shm[LINE_SIZE]; // does not depend on the seed
    __shared__ float inertia_shm[NUM_SEEDS]; // contains SSE for all seeds (but not rows because those are in blocks!)
    __shared__ float seed_centroids[NUM_SEEDS][NUM_CLUSTERS];
    __shared__ int seed_labels[NUM_SEEDS][LINE_SIZE];

    memcpy(input_shm, input + row * LINE_SIZE, sizeof(float) * LINE_SIZE); // TODO: check if correct: was input + row * LINE_SIZE + seed, got rid of seed

    struct core_params_t core_params;
    core_params.line = input_shm;
    core_params.line_size = LINE_SIZE;
    core_params.labels = seed_labels[seed];
    core_params.centroids = seed_centroids[seed];
    core_params.num_clusters = NUM_CLUSTERS;
    core_params.seed = seed;
    core_params.inertia = inertia_shm[seed];

    __syncthreads();
    if (seed == 0) {
        getInitialCentroids(core_params, seed_centroids);
    }
    __syncthreads();

    kmeans(core_params); // outputs

    // Wait for all the seeds and all rows to complete
    // and then find which seed yields the lowest inertia
    // and use the result from that seed
    inertia_shm[seed] = calcInertia(core_params);

    __syncthreads();
    // now that all seeds are completed,
    // do the rest of the work on a single thread for this block
    if (seed == 0) {
        int min_index = 0;
        find_min(inertia_shm, NUM_SEEDS, &min_index);
        // printf("%f ", inertia_shm[min_index]);

        // Copy labels and centroids to their respective destination
        int* labels_line = LABELS_LINE(outputLabels, row);
        float* centroids_line = CENTROIDS_LINE(outputCentroids, row);
        memcpy(labels_line, seed_labels[min_index], LINE_SIZE * sizeof(int));
        memcpy(centroids_line, seed_centroids[min_index], NUM_CLUSTERS * sizeof(float));
    }
}


void initInputData(float** input) {
    // Seed random number generator
    srand(1);

    // Declare and allocate a two-dimensional array
    float* sample_data = (float*)malloc(NUM_ROWS * LINE_SIZE * sizeof(float));

    // Fill the array with random numbers from 1 to 100
    for (int row = 0; row < NUM_ROWS; ++row) {
        //printf("ROW=%d: ", row);
        for (int i = 0; i < LINE_SIZE; ++i) {
            sample_data[row * LINE_SIZE + i] = (rand() / float(RAND_MAX)) * MAX_INPUT_VALUE;
            //printf("%.2f ", sample_data[row * LINE_SIZE + i]);
        }
        //printf("\n");
    }

    cudaMalloc(input, NUM_ROWS * LINE_SIZE * sizeof(float)); //allocate memory on device for labels
    cudaMemcpy(*input, sample_data, NUM_ROWS * LINE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(sample_data);
}

void GPUKmeans() {

    // init output tensor as a 3D array
    cudaExtent labelsExtent = make_cudaExtent(sizeof(int), LINE_SIZE, NUM_ROWS); //width x height x depth = amount of memory to allocate
    cudaPitchedPtr outputLabels; // create the pointer needed for the next call (cudaMalloc3D)
    cudaMalloc3D(&outputLabels, labelsExtent); // allocate memory on GPU

    // init output tensor as a 3D array
    cudaExtent centroidsExtent = make_cudaExtent(sizeof(float), NUM_CLUSTERS, NUM_ROWS); //width x height x depth = amount of memory to allocate
    cudaPitchedPtr outputCentroids; // create the pointer needed for the next call (cudaMalloc3D)
    cudaMalloc3D(&outputCentroids, centroidsExtent); // allocate memory on GPU


    clock_t start, end;
    float gpu_time_used;
    float sum_of_time = 0.0;
    for (int i = 0; i < TEST_ITERATIONS; i++) {

        // initalize input data
        float* inputData; // NUM_ROWS x LINE_SIZE TODO: ints?
        initInputData(&inputData);

        start = clock();
        cuda_kmeans << <NUM_ROWS, NUM_SEEDS >> > (inputData, outputLabels, outputCentroids);
        cudaDeviceSynchronize();
        end = clock();
        gpu_time_used = ((float)(end - start)) / CLOCKS_PER_SEC;
        printf("INSIDE iter %d, Time taken to run k-means algorithm: %f\n", i, gpu_time_used);
        sum_of_time += gpu_time_used;

        cudaFree(inputData);
    }

    printf("Time taken to run k-means algorithm with %d seeds on %d rows and %d columns: %f\n", NUM_SEEDS, NUM_ROWS, LINE_SIZE, sum_of_time);
    printf("Average runtime: %f", sum_of_time / TEST_ITERATIONS);

    cudaFree(outputLabels.ptr);
    cudaFree(outputCentroids.ptr);

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

int main() {
    GPUKmeans();
    return 0;
}