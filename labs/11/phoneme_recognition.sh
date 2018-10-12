SCRIPT="python3.6 ./phoneme_recognition.py"
DIMS="128 256 512 1024"
CELLS="GRU LSTM"

for cell in $CELLS
do
	for dim in $DIMS
	do
		CMD="$SCRIPT --rnn_cell $cell --rnn_cell_dim $dim"
		echo $CMD
		$CMD
	done
done
