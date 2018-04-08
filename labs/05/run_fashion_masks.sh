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
#"--cnn=CB-10-3-2-same,F,R-100 --epochs 30"
#"--cnn=CB-10-3-1-same,F,R-100 --epochs 30"
#"--cnn=CB-20-3-2-same,F,R-100 --epochs 30"
#"--cnn=CB-20-3-1-same,F,R-100 --epochs 30"
#"--cnn=CB-10-3-2-same,CB-10-3-2-same,F,R-100 --epochs 240"
#"--cnn=CB-10-3-1-same,CB-10-3-1-same,F,R-100 --epochs 240"
#"--cnn=CB-20-3-2-same,CB-20-3-2-same,F,R-100 --epochs 240"
#"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-100 --epochs 240"
#"--cnn=CB-64-5-1-valid,M-3-2,CB-64-5-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-64-3-1-valid,M-3-2,CB-64-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"         # <- best hyper-parameters found
#"--cnn=CB-64-5-1-valid,M-3-2,CB-64-5-1-valid,F,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-64-3-1-valid,M-3-2,CB-64-3-1-valid,F,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-64-5-1-valid,M-3-2,CB-64-5-1-valid,F,R-200,D-0.6,R-200,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-64-3-1-valid,M-3-2,CB-64-3-1-valid,F,R-200,D-0.6,R-200,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-64-5-1-valid,M-3-2,CB-64-5-1-valid,F,R-150,D-0.6,R-150,D-0.6,R-150,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-64-3-1-valid,M-3-2,CB-64-3-1-valid,F,R-150,D-0.6,R-150,D-0.6,R-150,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#"--cnn=CB-5-3-1-valid,CB-10-3-1-valid,M-3-2,CB-20-3-1-valid,CB-30-3-1-valid,F,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"
#
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_mask * loss_pred
#"--cnn=CB-64-3-1-valid,M-3-2,CB-64-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_mask * loss_pred
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_mask(MSE) * loss_pred
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = continuous IOU
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = IOU with MSE for correct_masks
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_class * loss_iou
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = class_cosine_distance * loss_iou
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred - loss_pred * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (1 / loss_pred) * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.01 / loss_pred) * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.1 / loss_pred) * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.5 / loss_pred) * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.75 / loss_pred) * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask
#"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.65 / loss_pred) * loss_mask
"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask
"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask
"--cnn=CB-30-3-1-valid,M-3-2,CB-30-3-1-valid,F,D-0.6,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask
"--cnn=CB-60-3-1-valid,M-3-2,CB-60-3-1-valid,F,R-400 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask
"--cnn=CB-60-3-1-valid,M-3-2,CB-60-3-1-valid,F,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask
"--cnn=CB-60-3-1-valid,M-3-2,CB-60-3-1-valid,F,D-0.6,R-400,D-0.6 --batch_size 64 --epochs 60 --learning_rate 0.01 --learning_rate_final 0.0025"  # loss = loss_pred + (0.7 / loss_pred) * loss_mask

# setups from `run_mnist_competition.py`
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
