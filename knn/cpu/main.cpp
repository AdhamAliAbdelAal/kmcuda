#include <iostream>
#include <map>
#include <math.h>
using namespace std;

long long k, n;
int dim;

bool read_data(string input_file, float *&data, int *&labels, float *&target)
{
    FILE *file = freopen(input_file.c_str(), "r", stdin);
    if (file == NULL)
    {
        cout << "Cannot open file " << input_file << endl;
        return false;
    }
    cout << "Reading data from " << input_file << endl;
    cin >> k >> n >> dim;
    cout << "k = " << k << ", n = " << n << ", dim = " << dim << endl;
    data = (float *)malloc(sizeof(float) * n * dim);
    labels = (int *)malloc(sizeof(int) * n);
    target = (float *)malloc(sizeof(float) * dim);

    for (int i = 0; i < dim; i++)
    {
        cin >> target[i];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            cin >> data[i * dim + j];
        }
        cin >> labels[i];
    }
    fclose(file);
    return true;
}

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
                dist1 += (output[i * dim + l] - target[l]) * (output[i * dim + l] - target[l]);
                dist2 += (output[j * dim + l] - target[l]) * (output[j * dim + l] - target[l]);
            }
            if (dist1 > dist2)
            {
                for (int l = 0; l < dim; l++)
                {
                    float temp = output[i * dim + l];
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
        float dist = 0;
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

bool write_data(string output_file, float *output)
{
    map<int, int> labelCount;
    FILE *file = freopen(output_file.c_str(), "w", stdout);
    if (file == NULL)
    {
        cout << "Cannot open file " << output_file << endl;
        return false;
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < dim; j++)
        {
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

void print_top(float *data, int n, float *target)
{
    cout << "Target data: ";
    for (int i = 0; i < dim; i++)
    {
        cout << target[i] << " ";
    }
    cout << endl;
    cout << "Top " << n << " data:" << endl;
    for (int i = 0; i < n; i++)
    {
        cout << "Data " << i << ": ";
        for (int j = 0; j < dim; j++)
        {
            cout << data[i * dim + j] << " ";
        }
        double distance = 0;
        for (int j = 0; j < dim; j++)
        {
            distance += (data[i * dim + j] - target[j]) * (data[i * dim + j] - target[j]);
        }
        cout << "Distance: " << distance;
        cout << endl;
    }
}

void knn(float *data, float *target, float *output)
{
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            output[i * dim + j] = data[i * dim + j];
        }
    }

    // Implement knn here
    for (int i = k; i < n; i++)
    {
        // Calculate distance
        float distance = 0;
        for (int j = 0; j < dim; j++)
        {
            distance += (data[i * dim + j] - target[j]) * (data[i * dim + j] - target[j]);
        }
        // Find the maximum distance
        float maxDistance = 0;
        int maxIndex = 0;
        for (int j = 0; j < k; j++)
        {
            float currentDistance = 0;
            for (int l = 0; l < dim; l++)
            {
                currentDistance += (output[j * dim + l] - target[l]) * (output[j * dim + l] - target[l]);
            }
            if (currentDistance > maxDistance)
            {
                maxDistance = currentDistance;
                maxIndex = j;
            }
        }
        if (distance < maxDistance)
        {
            for (int j = 0; j < dim; j++)
            {
                output[maxIndex * dim + j] = data[i * dim + j];
            }
        }
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

    // allocate memory for output
    float *output = (float *)malloc(sizeof(float) * k * dim);
    // store start time
    clock_t start = clock();

    knn(data, target, output);
    // store end time
    clock_t end = clock();
    // calculate elapsed time   
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_time << "s" << endl;
    // Print top k data
    bubbleSortResult(output, target);

    // Write output
    write_data(output_file, output);

    // Free memory
    free(data);
    free(labels);
    free(target);
    free(output);

    return 0;
}