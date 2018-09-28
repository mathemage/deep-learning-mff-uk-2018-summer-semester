#!/usr/bin/env bash

python3 ./sequence_classification_sol.py --rnn_cell=RNN --sequence_dim=1
python3 ./sequence_classification_sol.py --rnn_cell=GRU --sequence_dim=1
python3 ./sequence_classification_sol.py --rnn_cell=LSTM --sequence_dim=1

python3 ./sequence_classification_sol.py --rnn_cell=RNN  --sequence_dim=2
python3 ./sequence_classification_sol.py --rnn_cell=GRU  --sequence_dim=2
python3 ./sequence_classification_sol.py --rnn_cell=LSTM  --sequence_dim=2

python3 ./sequence_classification_sol.py --rnn_cell=RNN --sequence_dim=10
python3 ./sequence_classification_sol.py --rnn_cell=GRU  --sequence_dim=10
python3 ./sequence_classification_sol.py --rnn_cell=LSTM --sequence_dim=10

python3 ./sequence_classification_sol.py --rnn_cell=LSTM --hidden_layer=50 --rnn_cell_dim=30 --sequence_dim=30
python3 ./sequence_classification_sol.py --rnn_cell=LSTM --hidden_layer=50 --rnn_cell_dim=30 --sequence_dim=30 --clip_gradient=1

python3 ./sequence_classification_sol.py --rnn_cell=RNN --hidden_layer=50 --rnn_cell_dim=30 --sequence_dim=30
python3 ./sequence_classification_sol.py --rnn_cell=RNN --hidden_layer=50 --rnn_cell_dim=30 --sequence_dim=30 --clip_gradient=1

python3 ./sequence_classification_sol.py --rnn_cell=GRU --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30
python3 ./sequence_classification_sol.py --rnn_cell=GRU --hidden_layer=70 --rnn_cell_dim=30 --sequence_dim=30 --clip_gradient=1
