#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include "mnist.h"

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.1
#define BATCH_SIZE 500
#define NUM_EPOCHS 50
#define NUM_THREADS 16

__global__ void init_weights_biases_kernel(float *weights, float *bias, int m, int n, float limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    curandStateXORWOW_t state;
    curand_init(123456789 + idx, 1, 0, &state);
    curand_uniform(&state);

    bias[idx % m] = -limit + 2 * curand_uniform(&state) * limit;
    weights[idx] = -limit + 2 * curand_uniform(&state) * limit;
}

__global__ void transpose_kernel(float *matrix, float *matrix_t, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    int i = idx / n, j = idx % n;
    matrix_t[i + m * j] = matrix[idx];
}

__global__ void matrix_multiply_kernel(float *A, float *B, float *C, bool transposeA, bool transposeB, int m, int n, int p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * p) return;

    int i = idx / p, j = idx % p;
    float total = 0;

    if (!transposeA && !transposeB)
        for (int k = 0; k < n; ++k) total += A[i * n + k] * B[k * p + j];
    else if (transposeA)
        for (int k = 0; k < n; ++k) total += A[i + m * k] * B[k * p + j];
    else if (transposeB)
        for (int k = 0; k < n; ++k) total += A[i * n + k] * B[k + n * j];

    C[idx] = total;
}

__global__ void relu_kernel(float *hidden, float *bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    hidden[idx] += bias[idx % m];
    hidden[idx] = hidden[idx] > 0 ? hidden[idx] : 0;
}

__global__ void relu_derivative_kernel(float *hidden, float *delta_hidden, float *delta_bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    delta_hidden[idx] = hidden[idx] > 0 ? delta_hidden[idx] : 0;
    atomicAdd(&delta_bias[idx % m], delta_hidden[idx] / n);
}

__global__ void softmax_add_kernel(float *output, float *bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    output[idx] += bias[idx % m];
    output[idx] = expf(output[idx]);
}

__global__ void softmax_normalize_kernel(float *output, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float total = 0;
    for (int j = 0; j < m; ++j) total += output[idx + j * n];
    for (int j = 0; j < m; ++j) output[idx + j * n] /= total;
}

__global__ void softmax_derivative_kernel(float *output, int *target, float *delta_output, float *delta_bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    delta_output[idx] = output[idx] - target[idx];
    atomicAdd(&delta_bias[idx % m], delta_output[idx] / n);
}

__global__ void update_weights_biases_kernel(float *weight, float *bias, float *delta_weight, float *delta_bias, int m, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    bias[idx % m] -= LEARNING_RATE / BATCH_SIZE * delta_bias[idx % m];
    weight[idx] -= LEARNING_RATE / BATCH_SIZE * delta_weight[idx];
}

void prepare_mnist_data(mnist_data *data, int count, float **inputs, int **targets)
{
    *inputs = (float *)malloc(count * INPUT_SIZE * sizeof(float));
    *targets = (int *)malloc(count * sizeof(int));

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < count; ++i)
    {
        for (int j = 0; j < 28; ++j)
            for (int k = 0; k < 28; ++k)
                (*inputs)[i * INPUT_SIZE + j * 28 + k] = data[i].data[j][k];

        (*targets)[i] = data[i].label;
    }
}

void shuffle_mnist_data(int *indices, int count)
{
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < count; ++i)
    {
        int j = rand() % count;
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

void prepare_mnist_batch(int i, int count, float *inputs, int *targets, int *indices, float *batch_inputs, int *batch_targets)
{
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int j = 0; j < BATCH_SIZE; ++j)
    {
        if (i + j >= count) continue;

        int index = indices[i + j];
        float *example = &inputs[index * INPUT_SIZE];

        for (int k = 0; k < INPUT_SIZE; ++k)
            batch_inputs[j + k * BATCH_SIZE] = example[k];

        batch_targets[j + targets[index] * BATCH_SIZE] = 1;
    }
}

void init_weights_biases(float *w1, float *w2, float *w3, float *b1, float *b2, float *b3)
{
    int NBLOCKS, NTHREADS_PER_BLOCK = 256;
    float limit;

    limit = 1.0 / sqrt(INPUT_SIZE - 1);
    NBLOCKS = (HIDDEN1_SIZE * INPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    init_weights_biases_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w1, b1, HIDDEN1_SIZE, INPUT_SIZE, limit);
    cudaDeviceSynchronize();

    limit = 1.0 / sqrt(HIDDEN1_SIZE - 1);
    NBLOCKS = (HIDDEN2_SIZE * HIDDEN1_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    init_weights_biases_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w2, b2, HIDDEN2_SIZE, HIDDEN1_SIZE, limit);
    cudaDeviceSynchronize();

    limit = 1.0 / sqrt(HIDDEN2_SIZE - 1);
    NBLOCKS = (OUTPUT_SIZE * HIDDEN2_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    init_weights_biases_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w3, b3, OUTPUT_SIZE, HIDDEN2_SIZE, limit);
    cudaDeviceSynchronize();
}

void forward_propagate(float *input, float *hidden1, float *hidden2, float *output,
                       float *w1, float *w2, float *w3, float *b1, float *b2, float *b3,
                       int size, float *alpha, float* beta, cublasHandle_t handle)
{
    int NBLOCKS, NTHREADS_PER_BLOCK = 256;

#ifdef USE_CUBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, HIDDEN1_SIZE, INPUT_SIZE, alpha, input, size, w1, INPUT_SIZE, beta, hidden1, size);
#else
    NBLOCKS = (size * HIDDEN1_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w1, input, hidden1, false, false, HIDDEN1_SIZE, INPUT_SIZE, size);
    cudaDeviceSynchronize();
#endif

    NBLOCKS = (size * HIDDEN1_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    relu_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(hidden1, b1, HIDDEN1_SIZE, size);
    cudaDeviceSynchronize();

#ifdef USE_CUBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, HIDDEN2_SIZE, HIDDEN1_SIZE, alpha, hidden1, size, w2, HIDDEN1_SIZE, beta, hidden2, size);
#else
    NBLOCKS = (size * HIDDEN2_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w2, hidden1, hidden2, false, false, HIDDEN2_SIZE, HIDDEN1_SIZE, size);
    cudaDeviceSynchronize();
#endif

    NBLOCKS = (size * HIDDEN2_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    relu_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(hidden2, b2, HIDDEN2_SIZE, size);
    cudaDeviceSynchronize();

#ifdef USE_CUBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, OUTPUT_SIZE, HIDDEN2_SIZE, alpha, hidden2, size, w3, HIDDEN2_SIZE, beta, output, size);
#else
    NBLOCKS = (size * OUTPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w3, hidden2, output, false, false, OUTPUT_SIZE, HIDDEN2_SIZE, size);
    cudaDeviceSynchronize();
#endif

    NBLOCKS = (size * OUTPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    softmax_add_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(output, b3, OUTPUT_SIZE, size);
    cudaDeviceSynchronize();

    NBLOCKS = (size + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    softmax_normalize_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(output, OUTPUT_SIZE, size);
    cudaDeviceSynchronize();
}

void backward_propagate(float *input, float *hidden1, float *hidden2, float *output, int *target,
                        float *w1, float *w2, float *w3, float *b1, float *b2, float *b3,
                        float *delta_hidden1, float *delta_hidden2, float *delta_output,
                        float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3,
                        float *alpha, float* beta, cublasHandle_t handle)
{
    int NBLOCKS, NTHREADS_PER_BLOCK = 256;

    NBLOCKS = (OUTPUT_SIZE * BATCH_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    softmax_derivative_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(output, target, delta_output, db3, OUTPUT_SIZE, BATCH_SIZE);
    cudaDeviceSynchronize();

#ifdef USE_CUBLAS
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE, alpha, hidden2, BATCH_SIZE, delta_output, BATCH_SIZE, beta, dw3, HIDDEN2_SIZE);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, BATCH_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE, alpha, delta_output, BATCH_SIZE, w3, HIDDEN2_SIZE, beta, delta_hidden2, BATCH_SIZE);
#else
    NBLOCKS = (OUTPUT_SIZE * HIDDEN2_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(delta_output, hidden2, dw3, false, true, OUTPUT_SIZE, BATCH_SIZE, HIDDEN2_SIZE);
    cudaDeviceSynchronize();

    NBLOCKS = (HIDDEN2_SIZE * BATCH_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w3, delta_output, delta_hidden2, true, false, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    cudaDeviceSynchronize();
#endif

    NBLOCKS = (HIDDEN2_SIZE * BATCH_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    relu_derivative_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(hidden2, delta_hidden2, db2, HIDDEN2_SIZE, BATCH_SIZE);
    cudaDeviceSynchronize();

#ifdef USE_CUBLAS
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE, alpha, hidden1, BATCH_SIZE, delta_hidden2, BATCH_SIZE, beta, dw2, HIDDEN1_SIZE);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, BATCH_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, alpha, delta_hidden2, BATCH_SIZE, w2, HIDDEN1_SIZE, beta, delta_hidden1, BATCH_SIZE);
#else
    NBLOCKS = (HIDDEN2_SIZE * HIDDEN1_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(delta_hidden2, hidden1, dw2, false, true, HIDDEN2_SIZE, BATCH_SIZE, HIDDEN1_SIZE);
    cudaDeviceSynchronize();

    NBLOCKS = (HIDDEN1_SIZE * BATCH_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w2, delta_hidden2, delta_hidden1, true, false, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE);
    cudaDeviceSynchronize();
#endif

    NBLOCKS = (HIDDEN1_SIZE * BATCH_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    relu_derivative_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(hidden1, delta_hidden1, db1, HIDDEN1_SIZE, BATCH_SIZE);
    cudaDeviceSynchronize();

#ifdef USE_CUBLAS
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, INPUT_SIZE, HIDDEN1_SIZE, BATCH_SIZE, alpha, input, BATCH_SIZE, delta_hidden1, BATCH_SIZE, beta, dw1, INPUT_SIZE);
#else
    NBLOCKS = (HIDDEN1_SIZE * INPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    matrix_multiply_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(delta_hidden1, input, dw1, false, true, HIDDEN1_SIZE, BATCH_SIZE, INPUT_SIZE);
    cudaDeviceSynchronize();
#endif
}

void update_weights_biases(float *w1, float *w2, float *w3, float *b1, float *b2, float *b3,
                           float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3)
{
    int NBLOCKS, NTHREADS_PER_BLOCK = 256;

    NBLOCKS = (HIDDEN1_SIZE * INPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    update_weights_biases_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w1, b1, dw1, db1, HIDDEN1_SIZE, INPUT_SIZE);
    cudaDeviceSynchronize();

    NBLOCKS = (HIDDEN2_SIZE * HIDDEN1_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    update_weights_biases_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w2, b2, dw2, db2, HIDDEN2_SIZE, HIDDEN1_SIZE);
    cudaDeviceSynchronize();

    NBLOCKS = (OUTPUT_SIZE * HIDDEN2_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    update_weights_biases_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(w3, b3, dw3, db3, OUTPUT_SIZE, HIDDEN2_SIZE);
    cudaDeviceSynchronize();
}

float cross_entropy_loss(float *output, int *target)
{
    float loss = 0;

#pragma omp parallel for reduction(- : loss) num_threads(NUM_THREADS)
    for (int i = 0; i < 10000 * OUTPUT_SIZE; ++i)
        loss -= target[i] * log(output[i]);

    return loss;
}

int main(int argc, char *argv[])
{
    mnist_data *train_data, *test_data;
    unsigned int train_count, test_count, validation_count;
    float *train_inputs, *test_inputs, *validation_inputs;
    int *train_targets, *test_targets, *validation_targets;
    int NBLOCKS, NTHREADS_PER_BLOCK = 256;

    mnist_load("mnist-data/train-images-idx3-ubyte", "mnist-data/train-labels-idx1-ubyte", &train_data, &train_count);
    mnist_load("mnist-data/t10k-images-idx3-ubyte", "mnist-data/t10k-labels-idx1-ubyte", &test_data, &test_count);

    train_count = 50000;
    validation_count = 10000;

    prepare_mnist_data(train_data, train_count, &train_inputs, &train_targets);
    prepare_mnist_data(test_data, test_count, &test_inputs, &test_targets);
    prepare_mnist_data(&train_data[train_count], validation_count, &validation_inputs, &validation_targets);

    float input[BATCH_SIZE * INPUT_SIZE];
    float test_outputs[test_count * OUTPUT_SIZE];

    int validation_target[validation_count * OUTPUT_SIZE] = {0};
    float validation_outputs[validation_count * OUTPUT_SIZE];
    int *indices = (int *)malloc(train_count * sizeof(int));

#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for
        for (int i = 0; i < train_count; ++i)
            indices[i] = i;

#pragma omp for
        for (int i = 0; i < validation_count; ++i)
            validation_target[i + validation_targets[i] * validation_count] = 1;
    }

    float *d_input, *d_input_t, *d_hidden1, *d_hidden2, *d_output, *d_delta_hidden1, *d_delta_hidden2, *d_delta_output;
    float *d_w1, *d_w2, *d_w3, *d_b1, *d_b2, *d_b3, *d_dw1, *d_dw2, *d_dw3, *d_db1, *d_db2, *d_db3;
    int *d_target;

    cudaMalloc((void **)&d_w1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc((void **)&d_w2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_w3, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
    cudaMalloc((void **)&d_b1, HIDDEN1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_b2, HIDDEN2_SIZE * sizeof(float));
    cudaMalloc((void **)&d_b3, OUTPUT_SIZE * sizeof(float));

    cudaMalloc((void **)&d_input, test_count * INPUT_SIZE * sizeof(float));
    cudaMalloc((void **)&d_input_t, test_count * INPUT_SIZE * sizeof(float));
    cudaMalloc((void **)&d_hidden1, test_count * HIDDEN1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_hidden2, test_count * HIDDEN2_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output, test_count * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void **)&d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(int));

    cudaMalloc((void **)&d_dw1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc((void **)&d_dw2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_dw3, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
    cudaMalloc((void **)&d_db1, HIDDEN1_SIZE * sizeof(float));
    cudaMalloc((void **)&d_db2, HIDDEN2_SIZE * sizeof(float));
    cudaMalloc((void **)&d_db3, OUTPUT_SIZE * sizeof(float));

    cudaMalloc((void **)&d_delta_hidden1, HIDDEN1_SIZE * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&d_delta_hidden2, HIDDEN2_SIZE * BATCH_SIZE * sizeof(float));
    cudaMalloc((void **)&d_delta_output, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));

    cudaMemcpy(d_input, validation_inputs, validation_count * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    NBLOCKS = (validation_count * INPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    transpose_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(d_input, d_input_t, validation_count, INPUT_SIZE);
    cudaDeviceSynchronize();

    float alpha = 1, beta = 0;
    cublasHandle_t handle;
    // FILE *file = fopen("output.txt", "w");

#ifdef USE_CUBLAS
    cublasCreate(&handle);
#endif

#ifdef USE_CUBLAS
    printf("GPU CUBLAS\n\nLEARNING_RATE = %.2lf   BATCH_SIZE = %d   NUM_EPOCHS = %d\n\n", LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS);
#else
    printf("GPU NATIVE\n\nLEARNING_RATE = %.2lf   BATCH_SIZE = %d   NUM_EPOCHS = %d\n\n", LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS);
#endif

    float total_training_time = 0;
    init_weights_biases(d_w1, d_w2, d_w3, d_b1, d_b2, d_b3);

    for (int k = 0; k < NUM_EPOCHS; ++k)
    {
        double epoch_start = omp_get_wtime();
        shuffle_mnist_data(indices, train_count);

        for (int i = 0; i < train_count; i += BATCH_SIZE)
        {
            int target[BATCH_SIZE * OUTPUT_SIZE] = {0};

            prepare_mnist_batch(i, train_count, train_inputs, train_targets, indices, input, target);
            cudaMemcpy(d_input, input, INPUT_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, target, OUTPUT_SIZE * BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

            forward_propagate(d_input, d_hidden1, d_hidden2, d_output, d_w1, d_w2, d_w3, d_b1, d_b2, d_b3, BATCH_SIZE, &alpha, &beta, handle);

            backward_propagate(d_input, d_hidden1, d_hidden2, d_output, d_target, d_w1, d_w2, d_w3, d_b1, d_b2, d_b3, d_delta_hidden1, d_delta_hidden2, d_delta_output, d_dw1, d_dw2, d_dw3, d_db1, d_db2, d_db3, &alpha, &beta, handle);

            update_weights_biases(d_w1, d_w2, d_w3, d_b1, d_b2, d_b3, d_dw1, d_dw2, d_dw3, d_db1, d_db2, d_db3);
        }

        double epoch_stop = omp_get_wtime();
        double epoch_time = epoch_stop - epoch_start;
        total_training_time += epoch_time;

        forward_propagate(d_input_t, d_hidden1, d_hidden2, d_output, d_w1, d_w2, d_w3, d_b1, d_b2, d_b3, validation_count, &alpha, &beta, handle);
        cudaMemcpy(validation_outputs, d_output, validation_count * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        float loss = cross_entropy_loss(validation_outputs, validation_target);
        // fprintf(file, "%lf ", loss / validation_count);
    }

    // fclose(file);

    cudaFree(d_dw1);
    cudaFree(d_dw2);
    cudaFree(d_dw3);
    cudaFree(d_db1);
    cudaFree(d_db2);
    cudaFree(d_db3);

    cudaFree(d_target);
    cudaFree(d_delta_hidden1);
    cudaFree(d_delta_hidden2);
    cudaFree(d_delta_output);

    cudaMemcpy(d_input, test_inputs, test_count * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    NBLOCKS = (test_count * INPUT_SIZE + NTHREADS_PER_BLOCK - 1) / NTHREADS_PER_BLOCK;
    transpose_kernel<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(d_input, d_input_t, test_count, INPUT_SIZE);
    cudaDeviceSynchronize();

    int correct_predictions = 0;
    double inference_start = omp_get_wtime();

    forward_propagate(d_input_t, d_hidden1, d_hidden2, d_output, d_w1, d_w2, d_w3, d_b1, d_b2, d_b3, test_count, &alpha, &beta, handle);
    cudaMemcpy(test_outputs, d_output, test_count * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

#pragma omp parallel for reduction(+ : correct_predictions) num_threads(NUM_THREADS)
    for (int i = 0; i < test_count; ++i)
    {
        int predicted_label = 0;
        float max_prob = test_outputs[i];

        for (int j = 1; j < OUTPUT_SIZE; ++j)
            if (test_outputs[i + j * test_count] > max_prob)
            {
                max_prob = test_outputs[i + j * test_count];
                predicted_label = j;
            }

        if (predicted_label == test_targets[i]) correct_predictions++;
    }

    double inference_stop = omp_get_wtime();
    double total_inference_time = inference_stop - inference_start;

    printf("Success Rate         (%%) : %9.6lf\n", 100.0 * correct_predictions / test_count);
    printf("Average Grind Rate (sps) : %9.2lf\n", train_count / total_training_time * NUM_EPOCHS);
    printf("Total Training Time  (s) : %9.6lf\n", total_training_time);
    printf("Total Inference Time (s) : %9.6lf\n", total_inference_time);

#ifdef USE_CUBLAS
    cublasDestroy(handle);
#endif

    free(train_data);
    free(test_data);
    free(train_inputs);
    free(train_targets);
    free(test_inputs);
    free(test_targets);
    free(validation_inputs);
    free(validation_targets);
    free(indices);

    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_w3);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);

    cudaFree(d_input);
    cudaFree(d_input_t);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
