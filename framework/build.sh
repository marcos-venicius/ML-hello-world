#!/bin/sh

set -xe

clang -Wall -Wextra -o nn.out nn.c -lm
