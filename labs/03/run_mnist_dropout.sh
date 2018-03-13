#!/usr/bin/env bash

INTERPRETER=/usr/bin/python3
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/03/mnist_dropout.py

ARGS=(
0 0.3 0.5 0.6 0.8 0.9
)

for arg in "${ARGS[@]}"; do
    command="$INTERPRETER $SCRIPT --dropout ${arg}"
    echo ${command}
    ${command}
    echo
done
