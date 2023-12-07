# ML hello world

**Notice this code is only to educational purposes, do not use it to any type of production ready application**

The `main.c` file has a basic ML algorithm that can learn `OR`, `AND` and `NAND` operations.

This algorithms is not capable to learn `XOR` because `XOR` cannot be linearly separated.

So, to do this job, we need more neurons.

So, the `deep-learning.c` file has a "more complex" ML algorithm with 3 neurons and 9 parameters that can handle `OR`, `AND`, `NAND` and `XOR`.

## How to run?

Running `main.c`:

```shell
gcc main.c -lm -o main && ./main
```

Running `deep-learning.c`:

```shell
gcc deep-learning.c -lm -o deep-learning && ./deep-learning
```
