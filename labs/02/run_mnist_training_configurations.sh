#!/usr/bin/env bash

INTERPRETER=/usr/bin/python3
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/02/mnist_training.py

ARGS=(
"SGD  0.01"
"SGD  0.01 --momentum 0.9"
"SGD  0.1"
"Adam 0.001"
"Adam 0.01 --learning_rate_final 0.001"
)

for line in "${ARGS[@]}"; do
    arg=($line)
    command="$INTERPRETER $SCRIPT --optimizer ${arg[0]} --learning_rate ${arg[1]}"
    echo ${command}
    ${command}
    echo
done
