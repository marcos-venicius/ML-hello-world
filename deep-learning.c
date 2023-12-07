#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef float sample[3];

typedef struct
{
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} Xor;

float random_float()
{
    return (float)rand() / (float)(RAND_MAX);
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x1, float x2)
{
    float a = sigmoidf(m.or_w1 * x1 + m.or_w2 * x2 + m.or_b);
    float b = sigmoidf(m.nand_w1 * x1 + m.nand_w2 * x2 + m.nand_b);
    return sigmoidf(a * m.and_w1 + b * m.and_w2 + m.and_b);
}

sample xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}};

sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1}};

sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}};

sample or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}};

sample *train = or_train;
size_t train_count = 4;

float cost(Xor m)
{
    float result = 0.0f;

    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);

        float d = y - train[i][2];

        result += d * d;
    }

    return result / train_count;
}

Xor random_xor()
{
    Xor m;

    m.or_w1 = random_float();
    m.or_w2 = random_float();
    m.or_b = random_float();

    m.nand_w1 = random_float();
    m.nand_w2 = random_float();
    m.nand_b = random_float();

    m.and_w1 = random_float();
    m.and_w2 = random_float();
    m.and_b = random_float();

    return m;
}

void print_xor(Xor m)
{
    printf("or_w1 = %f\n", m.or_w1);
    printf("or_w2 = %f\n", m.or_w2);
    printf("or_b = %f\n", m.or_b);

    printf("nand_w1 = %f\n", m.nand_w1);
    printf("nand_w2 = %f\n", m.nand_w2);
    printf("nand_b = %f\n", m.nand_b);

    printf("and_w1 = %f\n", m.and_w1);
    printf("and_w2 = %f\n", m.and_w2);
    printf("and_b = %f\n", m.and_b);
}

Xor finite_diff(Xor m, float eps)
{
    Xor g;
    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    return g;
}

Xor learn(Xor m, Xor g, float rate)
{
    m.or_w1 -= rate * g.or_w1;
    m.or_w2 -= rate * g.or_w2;
    m.or_b -= rate * g.or_b;
    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b -= rate * g.nand_b;
    m.and_w1 -= rate * g.and_w1;
    m.and_w2 -= rate * g.and_w2;
    m.and_b -= rate * g.and_b;

    return m;
}

void predict(char* label, sample data[], char symbol)
{
    printf("\n%s:\n", label);

    train = data;

    Xor m = random_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for (size_t i = 0; i < 2000 * 1000; i++)
    {
        Xor g = finite_diff(m, eps);
        m = learn(m, g, rate);
    }

    printf("\ncost = %f\n\n", cost(m));

    printf("PREDICTIONS: \n\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu %c %zu = %f\n", i, symbol, j, forward(m, i, j));
        }
    }
}

int main(void)
{
    predict("And", and_train, '&');
    predict("Or", or_train, '|');
    predict("Xor", xor_train, '^');
    predict("Nand", nand_train, '~');

    return 0;
}
