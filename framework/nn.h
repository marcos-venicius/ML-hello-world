#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(x) sizeof((x)) / sizeof((x)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
    // pointer to the beginning of the array
    // this will be an array of floats and rows & cols with determine the shape of the matrix
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dest, Mat src);
void mat_fill(Mat m, float value);
void mat_dot(Mat dest, Mat a, Mat b); // the result of the "dot product" with be in matrix dest to avoice memory allocation
void mat_sum(Mat dest, Mat a);        // the result of the "sum" with be in matrix dest to avoice memory allocation
void mat_sig(Mat m);
void mat_print(Mat m, const char *name, size_t padding);

#define MAT_AT(matrix, row, col) (matrix).es[(row) * (matrix).stride + (col)]
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct
{
    size_t count;

    Mat *ws;
    Mat *bs;
    Mat *as; // the amount of activations is count+1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[nn.count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float epsilon, Mat train_input, Mat train_output);
void nn_learn(NN nn, NN g, float learning_rate);
void nn_train(NN nn, NN g, float learning_rate, float epsilon, Mat train_input, Mat train_output, size_t epochs);

#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols); // "*m.es" if i change the type of "es" i don't need to update this piece of code

    NN_ASSERT(m.es != NULL);

    return m;
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows * m.cols; i++)
        m.es[i] = rand_float() * (high - low) + low;
}

void mat_fill(Mat m, float value)
{
    for (size_t i = 0; i < m.rows * m.cols; i++)
        m.es[i] = value;
}

// the result of the "dot product" with be in matrix dest to avoice memory allocation
void mat_dot(Mat dest, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == b.cols);

    size_t n = a.cols;

    for (size_t row = 0; row < dest.rows; row++)
    {
        for (size_t col = 0; col < dest.cols; col++)
        {
            MAT_AT(dest, row, col) = 0;

            for (size_t i = 0; i < n; i++)
                MAT_AT(dest, row, col) += MAT_AT(a, row, i) * MAT_AT(b, i, col);
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dest, Mat src)
{
    NN_ASSERT(dest.rows == src.rows);
    NN_ASSERT(dest.cols == src.cols);

    for (size_t i = 0; i < dest.cols * dest.rows; i++)
        dest.es[i] = src.es[i];
}

// the result of the "sum" with be in matrix dest to avoice memory allocation
void mat_sum(Mat dest, Mat a)
{
    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == a.cols);

    for (size_t i = 0; i < dest.rows * dest.cols; i++)
        dest.es[i] += a.es[i];
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.cols * m.rows; i++)
        m.es[i] = sigmoidf(m.es[i]);
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);

    for (size_t row = 0; row < m.rows; row++)
    {
        printf("%*s    ", (int)padding, "");
        for (size_t col = 0; col < m.cols; col++)
            printf("%f ", MAT_AT(m, row, col));

        printf("\n");
    }

    printf("%*s]\n", (int)padding, "");
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;

    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);

    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);

    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[256];

    printf("\n%s = [\n", name);

    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws_%zu", i);

        mat_print(nn.ws[i], buf, 4);

        snprintf(buf, sizeof(buf), "bs_%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;

    float c = 0;

    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t output_cols = to.cols;

        for (size_t j = 0; j < output_cols; j++)
        {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);

            c += d * d;
        }
    }

    return c / n;
}

void nn_finite_diff(NN nn, NN g, float epsilon, Mat train_input, Mat train_output)
{
    float saved;

    float c = nn_cost(nn, train_input, train_output);

    for (size_t layer_index = 0; layer_index < nn.count; layer_index++)
    {
        for (size_t i = 0; i < nn.ws[layer_index].rows * nn.ws[layer_index].cols; i++)
        {
            saved = nn.ws[layer_index].es[i];
            nn.ws[layer_index].es[i] += epsilon;
            g.ws[layer_index].es[i] = (nn_cost(nn, train_input, train_output) - c) / epsilon;
            nn.ws[layer_index].es[i] = saved;
        }

        for (size_t i = 0; i < nn.bs[layer_index].rows * nn.bs[layer_index].cols; i++)
        {
            saved = nn.bs[layer_index].es[i];
            nn.bs[layer_index].es[i] += epsilon;
            g.bs[layer_index].es[i] = (nn_cost(nn, train_input, train_output) - c) / epsilon;
            nn.bs[layer_index].es[i] = saved;
        }
    }
}

void nn_learn(NN nn, NN g, float learning_rate)
{
    for (size_t layer = 0; layer < nn.count; layer++)
    {
        for (size_t i = 0; i < nn.ws[layer].rows * nn.ws[layer].cols; i++)
            nn.ws[layer].es[i] -= learning_rate * g.ws[layer].es[i];

        for (size_t i = 0; i < nn.bs[layer].rows * nn.bs[layer].cols; i++)
            nn.bs[layer].es[i] -= learning_rate * g.bs[layer].es[i];
    }
}

void nn_train(NN nn, NN g, float learning_rate, float epsilon, Mat train_input, Mat train_output, size_t epochs)
{
    for (size_t i = 0; i < epochs; i++)
    {
        nn_finite_diff(nn, g, epsilon, train_input, train_output);
        nn_learn(nn, g, learning_rate);
    }
}

#endif // NN_IMPLEMENTAION
