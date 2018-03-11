#!/usr/bin/env bash

INTERPRETER=/usr/bin/python3
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/01/mnist_layers_activations.py

#$COMMAND --layers 0 --activation none
#
#$COMMAND --layers 1 --activation none
#$COMMAND --layers 1 --activation sigmoid
#$COMMAND --layers 1 --activation tanh
#$COMMAND --layers 1 --activation relu
#
#$COMMAND --layers 3 --activation sigmoid
#$COMMAND --layers 3 --activation relu
#
#$COMMAND --layers 5 --activation sigmoid

ARGS=(
"0 none"
"1 none"
"1 sigmoid"
"1 tanh"
"1 relu"
"3 sigmoid"
"3 relu"
"5 sigmoid")

for line in "${ARGS[@]}"; do
    arg=($line)
    command="$INTERPRETER $SCRIPT --layers ${arg[0]} --activation ${arg[1]}"
    echo ${command}
    ${command}
    echo
done
