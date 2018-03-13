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
#95.97    mnist_training.py --optimizer=SGD --learning_rate=0.01
#97.99    mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9
#98.01    mnist_training.py --optimizer=SGD --learning_rate=0.1
#97.98    mnist_training.py --optimizer=Adam --learning_rate=0.001
#98.15    mnist_training.py --optimizer=Adam --learning_rate=0.01 --learning_rate_final=0.001

for line in "${ARGS[@]}"; do
    arg=($line)
    command="$INTERPRETER $SCRIPT --optimizer ${arg[0]} --learning_rate ${arg[1]} ${arg[2]} ${arg[3]}"
    echo ${command}
    ${command}
    echo
done
