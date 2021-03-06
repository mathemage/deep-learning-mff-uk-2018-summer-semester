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

INTERPRETER=/usr/bin/python3
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/04/mnist_conv.py

# "--batch_size", default=50, type=int, help="Batch size.")
# "--dropout", default=0.6, type=float, help="Dropout rate")
# "--cnn", default=None, type=str, help="Description of the CNN architecture.")
# "--epochs", default=10, type=int, help="Number of epochs.")
# "--threads", default=1, type=int, help="Maximum number of threads to use.")
ARGS=(
"--cnn=C-10-3-2-same,C-10-3-2-same,M-3-2,F,R-100"
)

for configuration in "${ARGS[@]}"; do
    command="$INTERPRETER $SCRIPT $configuration"
    for i in $(seq 1); do
        echo ${command}
        ${command}
        echo
    done
done
