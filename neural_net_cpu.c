#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <cblas.h>
#include "mnist.h"

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.1
#define BATCH_SIZE 500
#define NUM_EPOCHS 50
#define TILE_SIZE 16
#define NUM_THREADS 16

void prepare_mnist_data(mnist_data *data, int count, float **inputs, int **targets)
{
    *inputs = (float *)malloc(count * INPUT_SIZE * sizeof(float));
    *targets = (int *)malloc(count * sizeof(int));

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < 28; j++)
            for (int k = 0; k < 28; k++)
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

void transpose(float *matrix, float *matrix_t, int m, int n)
{
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            matrix_t[j * m + i] = matrix[i * n + j];
}

void matrix_multiply(float *A, float *B, float *C, bool transposeA, bool transposeB, int m, int n, int p)
{
#ifdef USE_CBLAS
    if (!transposeA && !transposeB)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, p, n, 1.0, A, n, B, p, 0.0, C, p);
    else if (transposeA)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, p, n, 1.0, A, m, B, p, 0.0, C, p);
    else if (transposeB)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, p, n, 1.0, A, n, B, n, 0.0, C, p);
#else
#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < p; ++j)
                C[i * p + j] = 0;

        if (!transposeA && !transposeB)
#pragma omp for collapse(2)
            for (int i = 0; i < m; i += TILE_SIZE)
                for (int j = 0; j < p; j += TILE_SIZE)
                    for (int k = 0; k < n; k += TILE_SIZE)
                        for (int ii = i; ii < i + TILE_SIZE && ii < m; ++ii)
                            for (int jj = j; jj < j + TILE_SIZE && jj < p; ++jj)
                            {
                                float total = 0;

                                for (int kk = k; kk < k + TILE_SIZE && kk < n; ++kk)
                                    total += A[ii * n + kk] * B[kk * p + jj];

                                C[ii * p + jj] += total;
                            }
        else if (transposeA)
#pragma omp for collapse(2)
            for (int i = 0; i < m; i += TILE_SIZE)
                for (int j = 0; j < p; j += TILE_SIZE)
                    for (int k = 0; k < n; k += TILE_SIZE)
                        for (int ii = i; ii < i + TILE_SIZE && ii < m; ++ii)
                            for (int jj = j; jj < j + TILE_SIZE && jj < p; ++jj)
                            {
                                float total = 0;

                                for (int kk = k; kk < k + TILE_SIZE && kk < n; ++kk)
                                    total += A[ii + m * kk] * B[kk * p + jj];

                                C[ii * p + jj] += total;
                            }
        else if (transposeB)
#pragma omp for collapse(2)
            for (int i = 0; i < m; i += TILE_SIZE)
                for (int j = 0; j < p; j += TILE_SIZE)
                    for (int k = 0; k < n; k += TILE_SIZE)
                        for (int ii = i; ii < i + TILE_SIZE && ii < m; ++ii)
                            for (int jj = j; jj < j + TILE_SIZE && jj < p; ++jj)
                            {
                                float total = 0;

                                for (int kk = k; kk < k + TILE_SIZE && kk < n; ++kk)
                                    total += A[ii * n + kk] * B[kk + n * jj];

                                C[ii * p + jj] += total;
                            }
    }
#endif
}

void init_weights_biases(float *w1, float *w2, float *w3, float *b1, float *b2, float *b3)
{
#pragma omp parallel num_threads(NUM_THREADS)
    {
        float limit = 1.0 / sqrt(INPUT_SIZE - 1);

#pragma omp for
        for (int i = 0; i < INPUT_SIZE; ++i)
        {
            b1[i] = -limit + 2 * limit * (float)rand() / RAND_MAX;
            for (int j = 0; j < HIDDEN1_SIZE; ++j)
                w1[i * HIDDEN1_SIZE + j] = -limit + 2 * limit * (float)rand() / RAND_MAX;
        }

        limit = 1.0 / sqrt(HIDDEN1_SIZE - 1);

#pragma omp for
        for (int i = 0; i < HIDDEN1_SIZE; ++i)
        {
            b2[i] = -limit + 2 * limit * (float)rand() / RAND_MAX;
            for (int j = 0; j < HIDDEN2_SIZE; ++j)
                w2[i * HIDDEN2_SIZE + j] = -limit + 2 * limit * (float)rand() / RAND_MAX;
        }

        limit = 1.0 / sqrt(HIDDEN2_SIZE - 1);

#pragma omp for
        for (int i = 0; i < HIDDEN2_SIZE; ++i)
        {
            b3[i] = -limit + 2 * limit * (float)rand() / RAND_MAX;
            for (int j = 0; j < OUTPUT_SIZE; ++j)
                w3[i * OUTPUT_SIZE + j] = -limit + 2 * limit * (float)rand() / RAND_MAX;
        }
    }
}

void forward_propagate(float *input, float *hidden1, float *hidden2, float *output,
                       float *w1, float *w2, float *w3, float *b1, float *b2, float *b3, int size)
{
    matrix_multiply(w1, input, hidden1, false, false, HIDDEN1_SIZE, INPUT_SIZE, size);

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < HIDDEN1_SIZE * size; ++i)
    {
        hidden1[i] += b1[i % HIDDEN1_SIZE];
        hidden1[i] = hidden1[i] > 0 ? hidden1[i] : 0;
    }

    matrix_multiply(w2, hidden1, hidden2, false, false, HIDDEN2_SIZE, HIDDEN1_SIZE, size);

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < HIDDEN2_SIZE * size; ++i)
    {
        hidden2[i] += b2[i % HIDDEN2_SIZE];
        hidden2[i] = hidden2[i] > 0 ? hidden2[i] : 0;
    }

    matrix_multiply(w3, hidden2, output, false, false, OUTPUT_SIZE, HIDDEN2_SIZE, size);

#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for
        for (int i = 0; i < OUTPUT_SIZE * size; ++i)
        {
            output[i] += b3[i % OUTPUT_SIZE];
            output[i] = exp(output[i]);
        }

#pragma omp for
        for (int i = 0; i < size; ++i)
        {
            float total = 0;

            for (int j = 0; j < OUTPUT_SIZE; ++j)
                total += output[i + j * size];

            for (int j = 0; j < OUTPUT_SIZE; ++j)
                output[i + j * size] /= total;
        }
    }
}

void backward_propagate(float *input, float *hidden1, float *hidden2, float *output, int *target,
                        float *w1, float *w2, float *w3, float *b1, float *b2, float *b3,
                        float *delta_hidden1, float *delta_hidden2, float *delta_output,
                        float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3)
{
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; ++i)
    {
        delta_output[i] = output[i] - target[i];
        db3[i % OUTPUT_SIZE] += delta_output[i] / BATCH_SIZE;
    }

    matrix_multiply(delta_output, hidden2, dw3, false, true, OUTPUT_SIZE, BATCH_SIZE, HIDDEN2_SIZE);
    matrix_multiply(w3, delta_output, delta_hidden2, true, false, HIDDEN2_SIZE, OUTPUT_SIZE, BATCH_SIZE);

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < HIDDEN2_SIZE * BATCH_SIZE; ++i)
    {
        delta_hidden2[i] = hidden2[i] > 0 ? delta_hidden2[i] : 0;
        db2[i % HIDDEN2_SIZE] += delta_hidden2[i] / BATCH_SIZE;
    }

    matrix_multiply(delta_hidden2, hidden1, dw2, false, true, HIDDEN2_SIZE, BATCH_SIZE, HIDDEN1_SIZE);
    matrix_multiply(w2, delta_hidden2, delta_hidden1, true, false, HIDDEN1_SIZE, HIDDEN2_SIZE, BATCH_SIZE);

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < HIDDEN1_SIZE * BATCH_SIZE; ++i)
    {
        delta_hidden1[i] = hidden1[i] > 0 ? delta_hidden1[i] : 0;
        db1[i % HIDDEN1_SIZE] += delta_hidden1[i] / BATCH_SIZE;
    }

    matrix_multiply(delta_hidden1, input, dw1, false, true, HIDDEN1_SIZE, BATCH_SIZE, INPUT_SIZE);
}

void update_weights_biases(float *w1, float *w2, float *w3, float *b1, float *b2, float *b3,
                           float *dw1, float *dw2, float *dw3, float *db1, float *db2, float *db3)
{
#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for
        for (int i = 0; i < HIDDEN1_SIZE; ++i)
        {
            b1[i] -= LEARNING_RATE / BATCH_SIZE * db1[i];
            for (int j = 0; j < INPUT_SIZE; ++j)
                w1[i * INPUT_SIZE + j] -= LEARNING_RATE / BATCH_SIZE * dw1[i * INPUT_SIZE + j];
        }

#pragma omp for
        for (int i = 0; i < HIDDEN2_SIZE; ++i)
        {
            b2[i] -= LEARNING_RATE / BATCH_SIZE * db2[i];
            for (int j = 0; j < HIDDEN1_SIZE; ++j)
                w2[i * HIDDEN1_SIZE + j] -= LEARNING_RATE / BATCH_SIZE * dw2[i * HIDDEN1_SIZE + j];
        }

#pragma omp for
        for (int i = 0; i < OUTPUT_SIZE; ++i)
        {
            b3[i] -= LEARNING_RATE / BATCH_SIZE * db3[i];
            for (int j = 0; j < HIDDEN2_SIZE; ++j)
                w3[i * HIDDEN2_SIZE + j] -= LEARNING_RATE / BATCH_SIZE * dw3[i * HIDDEN2_SIZE + j];
        }
    }
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

    mnist_load("mnist-data/train-images-idx3-ubyte", "mnist-data/train-labels-idx1-ubyte", &train_data, &train_count);
    mnist_load("mnist-data/t10k-images-idx3-ubyte", "mnist-data/t10k-labels-idx1-ubyte", &test_data, &test_count);

    train_count = 50000;
    validation_count = 10000;

    prepare_mnist_data(train_data, train_count, &train_inputs, &train_targets);
    prepare_mnist_data(test_data, test_count, &test_inputs, &test_targets);
    prepare_mnist_data(&train_data[train_count], validation_count, &validation_inputs, &validation_targets);

    int test_target[10000 * OUTPUT_SIZE] = {0};
    int *indices = (int *)malloc(train_count * sizeof(int));

#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for
        for (int i = 0; i < train_count; ++i)
            indices[i] = i;

#pragma omp for
        for (int i = 0; i < validation_count; ++i)
            test_target[i + validation_targets[i] * validation_count] = 1;
    }

    float w1[HIDDEN1_SIZE * INPUT_SIZE], w2[HIDDEN2_SIZE * HIDDEN1_SIZE], w3[OUTPUT_SIZE * HIDDEN2_SIZE];
    float b1[HIDDEN1_SIZE], b2[HIDDEN2_SIZE], b3[OUTPUT_SIZE];

    float input[INPUT_SIZE * BATCH_SIZE], hidden1[HIDDEN1_SIZE * BATCH_SIZE], hidden2[HIDDEN2_SIZE * BATCH_SIZE], output[OUTPUT_SIZE * BATCH_SIZE];
    float delta_output[OUTPUT_SIZE * BATCH_SIZE], delta_hidden2[HIDDEN2_SIZE * BATCH_SIZE], delta_hidden1[HIDDEN1_SIZE * BATCH_SIZE];
    float dw1[HIDDEN1_SIZE * INPUT_SIZE], dw2[HIDDEN2_SIZE * HIDDEN1_SIZE], dw3[OUTPUT_SIZE * HIDDEN2_SIZE];
    float db1[HIDDEN1_SIZE], db2[HIDDEN2_SIZE], db3[OUTPUT_SIZE];

    float *test_inputs_t = malloc(test_count * INPUT_SIZE * sizeof(float));
    float *test_hidden1 = malloc(test_count * HIDDEN1_SIZE * sizeof(float));
    float *test_hidden2 = malloc(test_count * HIDDEN2_SIZE * sizeof(float));
    float *test_outputs = malloc(test_count * OUTPUT_SIZE * sizeof(float));

    transpose(validation_inputs, test_inputs_t, validation_count, INPUT_SIZE);

    // FILE *file = fopen("output.txt", "w");

#ifdef USE_CBLAS
    printf("CPU BLAS\n\nLEARNING_RATE = %.2lf   BATCH_SIZE = %d   NUM_EPOCHS = %d\n\n", LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS);
#else
    printf("CPU NATIVE\n\nLEARNING_RATE = %.2lf   BATCH_SIZE = %d   NUM_EPOCHS = %d   TILE_SIZE = %d\n\n", LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, TILE_SIZE);
#endif

    float total_training_time = 0;
    init_weights_biases(w1, w2, w3, b1, b2, b3);

    for (int k = 0; k < NUM_EPOCHS; ++k)
    {
        double epoch_start = omp_get_wtime();
        shuffle_mnist_data(indices, train_count);

        for (int i = 0; i < train_count; i += BATCH_SIZE)
        {
            int target[BATCH_SIZE * OUTPUT_SIZE] = {0};

            prepare_mnist_batch(i, train_count, train_inputs, train_targets, indices, input, target);

            forward_propagate(input, hidden1, hidden2, output, w1, w2, w3, b1, b2, b3, BATCH_SIZE);

            backward_propagate(input, hidden1, hidden2, output, target, w1, w2, w3, b1, b2, b3, delta_hidden1, delta_hidden2, delta_output, dw1, dw2, dw3, db1, db2, db3);

            update_weights_biases(w1, w2, w3, b1, b2, b3, dw1, dw2, dw3, db1, db2, db3);
        }

        double epoch_stop = omp_get_wtime();
        double epoch_time = epoch_stop - epoch_start;
        total_training_time += epoch_time;

        forward_propagate(test_inputs_t, test_hidden1, test_hidden2, test_outputs, w1, w2, w3, b1, b2, b3, validation_count);
        float loss = cross_entropy_loss(test_outputs, test_target);
        // fprintf(file, "%lf ", loss / validation_count);
    }

    // fclose(file);

    transpose(test_inputs, test_inputs_t, test_count, INPUT_SIZE);

    int correct_predictions = 0;
    double inference_start = omp_get_wtime();

    forward_propagate(test_inputs_t, test_hidden1, test_hidden2, test_outputs, w1, w2, w3, b1, b2, b3, test_count);

#pragma omp parallel for reduction(+ : correct_predictions) num_threads(NUM_THREADS)
    for (int i = 0; i < test_count; i++)
    {
        int predicted_label = 0;
        float max_prob = test_outputs[i];

        for (int j = 1; j < OUTPUT_SIZE; j++)
            if (test_outputs[i + j * test_count] > max_prob)
            {
                max_prob = test_outputs[i + j * test_count];
                predicted_label = j;
            }

        if (predicted_label == test_targets[i])
            correct_predictions++;
    }

    double inference_stop = omp_get_wtime();
    double total_inference_time = inference_stop - inference_start;

    printf("Success Rate         (%%) : %9.6lf\n", 100.0 * correct_predictions / test_count);
    printf("Average Grind Rate (sps) : %9.2lf\n", train_count / total_training_time * NUM_EPOCHS);
    printf("Total Training Time  (s) : %9.6lf\n", total_training_time);
    printf("Total Inference Time (s) : %9.6lf\n", total_inference_time);

    free(train_data);
    free(test_data);
    free(train_inputs);
    free(train_targets);
    free(test_inputs);
    free(test_targets);
    free(validation_inputs);
    free(validation_targets);
    free(indices);

    free(test_inputs_t);
    free(test_hidden1);
    free(test_hidden2);
    free(test_outputs);

    return EXIT_SUCCESS;
}
