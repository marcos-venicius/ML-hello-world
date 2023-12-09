#define NN_IMPLEMENTATION

#include "nn.h"

int main(void)
{
    Mat a = mat_alloc(5, 5);
    Mat b = mat_alloc(5, 5);

    mat_fill(a, 1);
    mat_fill(b, 1);

    mat_print(a);
    mat_print(b);

    mat_sum(b, a);

    mat_print(b);

    Mat c = mat_alloc(5, 5);

    mat_dot(c, b, b);

    mat_print(c);

    return 0;
}
