#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float train[][3] = {
    { 0, 0, 0 },
    { 0, 1, 1 },
    { 1, 0, 1 },
    { 1, 1, 1 }
};

#define train_count sizeof(train) / sizeof(train[0])

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float random_float()
{
    return (float)rand() / (float)(RAND_MAX);
}

float cost(float w1, float w2, float b)
{
    float result = 0.0f;

    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1*w1 + x2*w2 + b);

        float d = y - train[i][2];

        result += d*d;
    }

    return result / train_count;
}

float predict(float w1, float w2, float n1, float n2)
{
    return n1*w1 + n2*w2;
}

int main(int argc, char** argv)
{
    srand(time(0));

    float w1 = random_float();
    float w2 = random_float();
    float b = random_float();

    float eps = 1e-3;
    float rate = 1e-1;
    size_t iterations = 2000*2000;

    for (size_t i = 0; i < iterations; i++)
    {
        float c = cost(w1, w2, b);

        float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        float db = (cost(w1, w2, b + eps) - c) / eps;

        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;

    }

    printf("Train count: %ld\n", iterations);
    printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, cost(w1, w2, b));
    printf("\nPERFORMANCE:\n\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            int expected = (int)((float)(j + i) / 2 + 0.5f);

            printf("%zu | %zu = %f, expected = %d\n", i, j, sigmoidf(i*w1 + j*w2 + b), expected);
        }
    }

    return 0;
}
