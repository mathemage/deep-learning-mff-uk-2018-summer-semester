#!/usr/bin/env bash

python3 ./tagger_we_sol.py --rnn_cell=RNN --rnn_cell_dim=5
python3 ./tagger_we_sol.py --rnn_cell=RNN --rnn_cell_dim=10
python3 ./tagger_we_sol.py --rnn_cell=RNN --rnn_cell_dim=50

python3 ./tagger_we_sol.py --rnn_cell=GRU --rnn_cell_dim=5
python3 ./tagger_we_sol.py --rnn_cell=GRU --rnn_cell_dim=10
python3 ./tagger_we_sol.py --rnn_cell=GRU --rnn_cell_dim=50

python3 ./tagger_we_sol.py --rnn_cell=LSTM --rnn_cell_dim=5
python3 ./tagger_we_sol.py --rnn_cell=LSTM --rnn_cell_dim=10
python3 ./tagger_we_sol.py --rnn_cell=LSTM --rnn_cell_dim=50
