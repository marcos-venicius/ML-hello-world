#define NN_IMPLEMENTATION

#include "nn.h"
#include <time.h>

typedef struct
{
    Mat a0, a1, a2;

    // layer one
    Mat w1, b1;

    // layer two
    Mat w2, b2;
} Xor;

Xor xor_alloc(void)
{
    Xor m;

    m.a0 = mat_alloc(1, 2);

    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);

    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);

    return m;
}

void forward_xor(Xor m)
{
    // sigmoidf(x * w1 + b1)
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    // sigmoidf(a1 * w2 + b2)
    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
}

float cost(Xor m, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);

    size_t n = ti.rows;

    float c = 0;

    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);

        forward_xor(m);

        size_t q = to.cols;

        for (size_t j = 0; j < q; j++)
        {
            float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);

            c += d * d;
        }
    }

    return c / n;
}

void finite_diff(Xor model, Xor gradient, float epsilon, Mat ti, Mat to)
{
    float saved;

    float c = cost(model, ti, to);

    for (size_t i = 0; i < model.w1.rows * model.w1.cols; i++)
    {
        saved = model.w1.es[i];
        model.w1.es[i] += epsilon;
        gradient.w1.es[i] = (cost(model, ti, to) - c) / epsilon;
        model.w1.es[i] = saved;
    }

    for (size_t i = 0; i < model.b1.rows * model.b1.cols; i++)
    {
        saved = model.b1.es[i];
        model.b1.es[i] += epsilon;
        gradient.b1.es[i] = (cost(model, ti, to) - c) / epsilon;
        model.b1.es[i] = saved;
    }

    for (size_t i = 0; i < model.w2.rows * model.w2.cols; i++)
    {
        saved = model.w2.es[i];
        model.w2.es[i] += epsilon;
        gradient.w2.es[i] = (cost(model, ti, to) - c) / epsilon;
        model.w2.es[i] = saved;
    }

    for (size_t i = 0; i < model.b2.rows * model.b2.cols; i++)
    {
        saved = model.b2.es[i];
        model.b2.es[i] += epsilon;
        gradient.b2.es[i] = (cost(model, ti, to) - c) / epsilon;
        model.b2.es[i] = saved;
    }
}

// learn applying the gradient with a learning rate
void xor_learn(Xor model, Xor gradient, float learning_rate)
{
    for (size_t i = 0; i < model.w1.rows * model.w1.cols; i++)
        model.w1.es[i] -= learning_rate * gradient.w1.es[i];

    for (size_t i = 0; i < model.b1.rows * model.b1.cols; i++)
        model.b1.es[i] -= learning_rate * gradient.b1.es[i];

    for (size_t i = 0; i < model.w2.rows * model.w2.cols; i++)
        model.w2.es[i] -= learning_rate * gradient.w2.es[i];

    for (size_t i = 0; i < model.b2.rows * model.b2.cols; i++)
        model.b2.es[i] -= learning_rate * gradient.b2.es[i];
}

void train(Xor model, Xor gradient, Mat ti, Mat to, size_t epochs)
{
    float epsilon = 1e-1;
    float learning_rate = 1e-1;

    for (size_t i = 0; i < epochs; i++)
    {
        finite_diff(model, gradient, epsilon, ti, to);
        xor_learn(model, gradient, learning_rate);
    }
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

int main(void)
{
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td};

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + stride // point to the last element
    };

    Xor model = xor_alloc();
    Xor gradient = xor_alloc();

    // initialize matrices
    mat_rand(model.w1, 0, 1);
    mat_rand(model.b1, 0, 1);
    mat_rand(model.w2, 0, 1);
    mat_rand(model.b2, 0, 1);

    printf("\ncost = %f\n", cost(model, ti, to));

    MAT_PRINT(model.w1);
    MAT_PRINT(model.b1);
    MAT_PRINT(model.a1);

    MAT_PRINT(model.w2);
    MAT_PRINT(model.b2);
    MAT_PRINT(model.a2);

    size_t epochs = 1000 * 1000;

    printf("\n[*] Start training with %ld epochs...\n", epochs);

    train(model, gradient, ti, to, epochs);

    printf("\n[*] Training finished!\n");

    printf("\ncost = %f\n", cost(model, ti, to));

    MAT_PRINT(model.w1);
    MAT_PRINT(model.b1);
    MAT_PRINT(model.a1);

    MAT_PRINT(model.w2);
    MAT_PRINT(model.b2);
    MAT_PRINT(model.a2);

    printf("\nShowing results:\n\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            // load inputs
            MAT_AT(model.a0, 0, 0) = i;
            MAT_AT(model.a0, 0, 1) = j;

            forward_xor(model);

            float y = *model.a2.es;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    return 0;
}
