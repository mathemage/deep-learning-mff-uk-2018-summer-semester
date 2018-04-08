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
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/05/fashion_masks.py

#"--batch_size", default=2048, type=int, help="Batch size.")
#"--epochs", default=120, type=int, help="Number of epochs.")
#"--cnn", default="CB-10-3-2-same,M-3-2,F,R-100", type=str, help="Description of the CNN architecture.")
ARGS=(
"--cnn=CB-10-3-2-same,F,R-100 --epochs 10"
"--cnn=CB-10-3-1-same,F,R-100 --epochs 10"
"--cnn=CB-20-3-2-same,F,R-100 --epochs 10"
"--cnn=CB-20-3-1-same,F,R-100 --epochs 10"

#"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256"
#"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256"
#"--cnn=CB-20-3-1-same,M-2-1,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256"
#"--cnn=CB-20-3-1-same,M-3-2,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256" # <- best & submitted
#"--cnn=CB-10-3-1-same,M-2-1,CB-10-3-1-same,M-2-1,CB-10-3-1-same,F,R-300 --epochs  78 --batch_size 256"
#"--cnn=CB-10-3-1-same,CB-10-3-1-same,CB-10-3-1-same,F,R-300 --epochs  78 --batch_size 256"
#"--cnn=CB-10-3-1-same,CB-10-3-1-same,CB-10-3-1-same,M-2-1,F,R-300 --epochs  78 --batch_size 256"
#"--cnn=CB-10-3-1-same,CB-10-3-1-same,CB-10-3-1-same,M-3-2,F,R-300 --epochs  78 --batch_size 256"
)

for configuration in "${ARGS[@]}"; do
    command="$INTERPRETER $SCRIPT $configuration"
    for i in $(seq 1); do
        echo ${command}
        ${command}
        echo
    done
done
