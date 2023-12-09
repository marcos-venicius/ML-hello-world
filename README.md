# ML hello world

**Notice this code is only to educational purposes, do not use it to any type of production ready application**

The `main.c` file has a basic ML algorithm that can learn `OR`, `AND` and `NAND` operations.

This algorithms is not capable to learn `XOR` because `XOR` cannot be linearly separated.

So, to do this job, we need more neurons.

So, the `deep-learning.c` file has a "more complex" ML algorithm with 3 neurons and 9 parameters that can handle `OR`, `AND`, `NAND` and `XOR`.

We also have the `dynamic-deep-learning.c`, that is a "improved" neural network that you can change the number of neurons dynamically by changing `NUMBER_OF_NEURONS`.

Another improvement is that in `dynamic-deep-learning.c` we don't need to `forward` the neural network mannualy, we do this by iteration over the network.

## How to run?

Running `main.c`:

```shell
gcc main.c -lm -o main && ./main
```

Output demonstration:

![image](https://github.com/marcos-venicius/ML-hello-world/assets/94018427/5d0aef19-9439-4717-9147-9aea26b308c2)

Running `deep-learning.c`:

```shell
gcc deep-learning.c -lm -o deep-learning && ./deep-learning
```

Output demonstration:

![image](https://github.com/marcos-venicius/ML-hello-world/assets/94018427/80b88e85-910c-4c4f-bfbe-528f66eb1776)

Running `dynamic-deep-learning.c`:

```shell
gcc dynamic-deep-learning.c -lm -o dynamic-deep-learning && ./dynamic-deep-learning
```

Output demonstration:

![image](https://github.com/marcos-venicius/ML-hello-world/assets/94018427/5cbb37f5-9b5f-4eaa-ac1a-cb9c8576c3ca)

