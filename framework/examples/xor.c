#define NN_IMPLEMENTATION

#include "../nn.h"
#include <time.h>

float td[] = {
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
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

    size_t arch[] = {2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);

    float epsilon = 1e-1;
    float learning_rate = 1e-1;
    size_t epochs = 1000 * 1000;

    printf("[*] Training... %ld epochs...\n", epochs);

    nn_train(nn, g, learning_rate, epsilon, ti, to, epochs);

    printf("[*] Trained\n");

    NN_PRINT(nn);

    printf("\nShowing results:\n\n");

    for (size_t input_index = 0; input_index < n; input_index++)
    {
        Mat input = mat_row(ti, input_index);

        mat_copy(NN_INPUT(nn), input);

        nn_forward(nn);

        Mat output = NN_OUTPUT(nn);

        printf("%f ^ %f = %f\n", input.es[0], input.es[1], output.es[0]);
    }

    return 0;
}

