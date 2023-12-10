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

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
    // pointer to the beginning of the array
    // this will be an array of floats and rows & cols with determine the shape of the matrix
} Mat;

#define MAT_AT(matrix, row, col) (matrix).es[(row) * (matrix).stride + (col)]
#define MAT_PRINT(m) mat_print(m, #m)

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dest, Mat src);
void mat_fill(Mat m, float value);
void mat_dot(Mat dest, Mat a, Mat b); // the result of the "dot product" with be in matrix dest to avoice memory allocation
void mat_sum(Mat dest, Mat a);        // the result of the "sum" with be in matrix dest to avoice memory allocation
void mat_sig(Mat m);
void mat_print(Mat m, const char *name);

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
    return (Mat)
    {
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

void mat_print(Mat m, const char *name)
{
    printf("\n%s = [\n", name);
    for (size_t row = 0; row < m.rows; row++)
    {
        for (size_t col = 0; col < m.cols; col++)
        {
            printf("  %f ", MAT_AT(m, row, col));
        }
        printf("\n");
    }
    printf("]\n");
}

#endif // NN_IMPLEMENTAION
