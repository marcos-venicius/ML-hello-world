#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NUMBER_OF_NEURONS 3

typedef float sample[3];
typedef struct
{
    float w1;
    float w2;
    float b;
} Neuron;

float random_float()
{
    return (float)rand() / (float)(RAND_MAX);
}

Neuron *create_neural_network()
{
    Neuron *neural_network = malloc(sizeof(Neuron) * NUMBER_OF_NEURONS);

    for (int i = 0; i < NUMBER_OF_NEURONS; i++)
    {
        Neuron neuron;

        neuron.w1 = random_float();
        neuron.w2 = random_float();
        neuron.b = random_float();

        neural_network[i] = neuron;
    }

    return neural_network;
}

void print_neuron(Neuron neuron)
{
    printf("W1 = %f, W2 = %f, b = %f\n", neuron.w1, neuron.w2, neuron.b);
}

float activate(float x)
{
    return 1.f / (1.f + expf(-x));
}

float forward(Neuron *neurons, float x1, float x2)
{
    float output = 0.0;

    for (int i = 0; i < NUMBER_OF_NEURONS; i++)
    {
        float weighted_sum = neurons[i].w1 * x1 + neurons[i].w2 * x2 + neurons[i].b;

        float activated_output = activate(weighted_sum);

        output += activated_output;
    }

    return output;
}

float cost(Neuron *neurons, sample train[])
{
    float result = 0.0f;

    for (size_t i = 0; i < 4; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(neurons, x1, x2);

        float d = y - train[i][2];

        result += d * d;
    }

    return result / 4;
}

Neuron *finite_diff(Neuron *neural_network, sample train[], float eps)
{
    Neuron *new_neural_network = malloc(sizeof(Neuron) * NUMBER_OF_NEURONS);

    float c = cost(neural_network, train);

    for (int i = 0; i < NUMBER_OF_NEURONS; i++)
    {
        Neuron neuron;
        float saved = neural_network[i].w1;

        neural_network[i].w1 += eps;
        neuron.w1 = (cost(neural_network, train) - c) / eps;
        neural_network[i].w1 = saved;

        saved = neural_network[i].w2;

        neural_network[i].w2 += eps;
        neuron.w2 = (cost(neural_network, train) - c) / eps;
        neural_network[i].w2 = saved;

        saved = neural_network[i].b;

        neural_network[i].b += eps;
        neuron.b = (cost(neural_network, train) - c) / eps;
        neural_network[i].b = saved;

        new_neural_network[i] = neuron;
    }

    return new_neural_network;
}

Neuron *learn(Neuron *neural_network, Neuron *new_neural_network, float rate)
{
    for (int i = 0; i < NUMBER_OF_NEURONS; i++)
    {
        neural_network[i].w1 -= rate * new_neural_network[i].w1;
        neural_network[i].w2 -= rate * new_neural_network[i].w2;
        neural_network[i].b -= rate * new_neural_network[i].b;
    }

    return neural_network;
}

Neuron *train_model(sample train_data[])
{
    Neuron *neurons = create_neural_network();

    float eps = 1e-1;
    float rate = 1e-1;

    for (size_t i = 0; i < 1000 * 1000; i++)
    {
        Neuron *g = finite_diff(neurons, train_data, eps);
        neurons = learn(neurons, g, rate);
    }

    return neurons;
}

void predict(char* label, char* symbol, Neuron *neural_network, sample data[])
{
    printf("\n%s:\n\n", label);

    printf("cost = %f\n", cost(neural_network, data));

    printf("\nNeurons:\n\n");

    for (int i = 0; i < NUMBER_OF_NEURONS; i++)
    {
        printf("Neuron %d: ", i);
        print_neuron(neural_network[i]);
    }

    printf("\nPredictions:\n\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu %s %zu = %f\n", i, symbol, j, forward(neural_network, i, j));
        }
    }
}

int main(void)
{
    sample and_train[] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 1}};

    sample or_train[] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1}};

    sample nand_train[] = {
        {0, 0, 1},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0}};

    sample xor_train[] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0}};

    sample nor_train[] = {
        {0, 0, 1},
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 0}};

    Neuron *and_neural_network_trained = train_model(and_train);

    predict("AND operator", "&", and_neural_network_trained, and_train);

    Neuron *or_neural_network_trained = train_model(or_train);

    predict("OR operator", "|", or_neural_network_trained, or_train);

    Neuron *nand_neural_network_trained = train_model(nand_train);

    predict("NAND operator", "~&", nand_neural_network_trained, nand_train);

    Neuron *xor_neural_network_trained = train_model(xor_train);

    predict("XOR operator", "^", xor_neural_network_trained, xor_train);

    Neuron *nor_neural_network_trained = train_model(nor_train);

    predict("NOR operator", "~|", nor_neural_network_trained, nor_train);

    return 0;
}
