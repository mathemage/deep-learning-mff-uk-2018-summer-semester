#!/usr/bin/env bash
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx ids anywhere
# in the first comment block in the source file, i.e., in the first
# consecutive range of lines beginning with `#`.
#
# You can find out ReCodEx id on URL when watching ReCodEx profile.
# The id has the following format: 01234567-89ab-cdef-0123-456789abcdef.
#
# c6ef657e-2082-11e8-9de3-00505601122b (Anastasia Lebedeva)
# 08a323e8-21f3-11e8-9de3-00505601122b (Karel Ha)
#
# Note that:
# - after running this script which
#   - which trains a neural net
#   - and evaluates it to infer caps on test data
# there was a postprocessing. We have:
# - opened the test data file in Vim
# - Ctrl-V -> Shift-G (to visual-block select of each line's first character)
# - Shift-U (to uppercase characters under selection).
# This should work since every line should start with a capital letter (and at least the 1st character in the 1st line
# was deemed by our neural net to be capitalized).
#
# It's a hack though.


INTERPRETER=/usr/bin/python3
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/03/uppercase.py

# "--activation", default="relu", type=str, help="Activation function.")
# "--alphabet_size", default=100, type=int, help="Alphabet size.")
# "--batch_size", default=256, type=int, help="Batch size.")
# "--dropout", default=0.6, type=float, help="Dropout rate")
# "--epochs", default=10, type=int, help="Number of epochs.")
# "--hidden_layer", default=20, type=int, help="Size of the hidden layer.")
# "--layers", default=1, type=int, help="Number of layers.")
# "--learning_rate", default=0.01, type=float, help="Initial learning rate.")
# "--learning_rate_final", default=0.001, type=float, help="Final learning rate.")
# "--momentum", default=None, type=float, help="Momentum.")
# "--optimizer", default="Adam", type=str, help="Optimizer to use.")
# "--threads", default=1, type=int, help="Maximum number of threads to use.")
# "--window", default=10, type=int, help="Size of the window to use.")

ARGS=(
#"relu 100 256 0.6 10 20 1 0.01 0.001 Adam"  # with and without 1-hot enc
#"relu 100 1024 0.6 10 20 1 0.01 0.001 Adam"  # with 1-hot enc
#"relu 100 1024 0.6 10 10 2 0.01 0.001 Adam"  # with 1-hot enc
#"relu 100 1024 0.6 10 20 2 0.01 0.001 Adam"  # with 1-hot enc
#"relu 100 2048 0.6 10 20 2 0.01 0.001 Adam"  # with 1-hot enc
#"relu 100 2048 0.6 10 100 2 0.01 0.001 Adam"  # with 1-hot enc
#"relu 100 2048 0.6 10 200 2 0.01 0.001 Adam"  # with 1-hot enc
"relu 100 2048 0.6 30 200 2 0.01 0.001 Adam"  # with 1-hot enc <- best
#"relu 100 2048 0.6 10 100 3 0.01 0.001 Adam"  # with 1-hot enc
)

for line in "${ARGS[@]}"; do
    arg=($line)
    hyperparams=""
    hyperparams="$hyperparams --activation ${arg[0]}"
    hyperparams="$hyperparams --alphabet_size ${arg[1]}"
    hyperparams="$hyperparams --batch_size ${arg[2]}"
    hyperparams="$hyperparams --dropout ${arg[3]}"
    hyperparams="$hyperparams --epochs ${arg[4]}"
    hyperparams="$hyperparams --hidden_layer ${arg[5]}"
    hyperparams="$hyperparams --layers ${arg[6]}"
    hyperparams="$hyperparams --learning_rate ${arg[7]}"
    hyperparams="$hyperparams --learning_rate_final ${arg[8]}"
    hyperparams="$hyperparams --optimize ${arg[9]}"
#	"--momentum", default=None, type=float, help="Momentum.")
#    "--threads", default=1, type=int, help="Maximum number of threads to use.")
#    "--window", default=10, type=int, help="Size of the window to use.")
    hyperparams="$hyperparams ${arg[10]}"
    command="$INTERPRETER $SCRIPT $hyperparams"
    for i in $(seq 1); do
        echo ${command}
        ${command}
        echo
    done
done
