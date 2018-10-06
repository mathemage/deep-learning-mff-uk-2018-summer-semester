#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import timit_mfcc26_dataset


class Network:
	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                             intra_op_parallelism_threads=threads))

	def construct(self, args, num_phones, mfcc_dim):
		with self.session.graph.as_default():
			# Inputs
			self.mfcc_lens = tf.placeholder(tf.int32, [None])
			self.mfccs = tf.placeholder(tf.float32, [None, None, mfcc_dim])
			self.phone_lens = tf.placeholder(tf.int32, [None])
			self.phones = tf.placeholder(tf.int32, [None, None])

			# Done: Convert the input phoneme sequences into sparse representation (tf.where and tf.gather_nd are useful).
			def get_sparse_tensor_from_dense(dense_tensor):
				nonzero_indices = tf.where(tf.not_equal(dense_tensor, 0))
				nonzero_values = tf.gather_nd(dense_tensor, nonzero_indices)
				return tf.SparseTensor(
					indices=nonzero_indices,
					values=nonzero_values,
					dense_shape=tf.shape(dense_tensor, out_type=tf.int64),
				)

			self.sparse_phones = get_sparse_tensor_from_dense(self.phones)

			# RNN Cell
			if args.rnn_cell == "LSTM":
				rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
			elif args.rnn_cell == "GRU":
				rnn_cell = tf.nn.rnn_cell.GRUCell
			else:
				raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

			# Done: Use a bidirectional RNN and an output linear layer without activation.
			inputs = self.mfccs
			(hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
				rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
				inputs, sequence_length=self.mfcc_lens, dtype=tf.float32)
			hidden_layer = tf.concat([hidden_layer_fwd, hidden_layer_bwd], axis=2)
			output_layer = tf.layers.dense(hidden_layer, num_phones + 1)
			logits = tf.transpose(output_layer, (1, 0, 2))  # to allow for `time_major == True`

			# Done: Utilize CTC loss (tf.nn.ctc_loss).
			# Done: - `losses`: vector of losses, with an element for each example in the batch
			losses = tf.nn.ctc_loss(
				labels=self.sparse_phones,
				inputs=logits,
				sequence_length=self.mfcc_lens
			)

			# Done: Perform decoding by a CTC decoder (either greedily using tf.nn.ctc_greedy_decoder, or with beam search
			#  employing tf.nn.ctc_beam_search_decoder).
			# TODO try instead
			# self.predictions, _ = tf.nn.ctc_beam_search_decoder(
			self.predictions, _ = tf.nn.ctc_greedy_decoder(
				inputs=logits,
				sequence_length=self.mfcc_lens
			)

			# Done: Evaluate results using normalized edit distance (tf.edit_distance).
			# Done:: - `edit_distances`: vector of edit distances, with an element for each batch example
			edit_distances = tf.edit_distance(
				hypothesis=tf.cast(self.predictions[0], tf.int32),
				truth=self.sparse_phones,
				normalize=True
			)

			# Training
			global_step = tf.train.create_global_step()
			self.training = tf.train.AdamOptimizer().minimize(tf.reduce_mean(losses), global_step=global_step,
			                                                  name="training")

			# Summaries
			self.current_edit_distance, self.update_edit_distance = tf.metrics.mean(edit_distances)
			self.current_loss, self.update_loss = tf.metrics.mean(losses)
			self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

			summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
			self.summaries = {}
			with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
				self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
				                           tf.contrib.summary.scalar("train/edit_distance", self.update_edit_distance)]
			with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for dataset in ["dev", "test"]:
					self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
					                           tf.contrib.summary.scalar(dataset + "/edit_distance", self.current_edit_distance)]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train_epoch(self, train, batch_size):
		while not train.epoch_finished():
			mfcc_lens, mfccs, phone_lens, phones = train.next_batch(batch_size)
			self.session.run(self.reset_metrics)
			self.session.run([self.training, self.summaries["train"]],
			                 {self.mfcc_lens : mfcc_lens, self.mfccs: mfccs,
			                  self.phone_lens: phone_lens, self.phones: phones})

	def evaluate(self, dataset_name, dataset, batch_size):
		self.session.run(self.reset_metrics)
		while not dataset.epoch_finished():
			mfcc_lens, mfccs, phone_lens, phones = dataset.next_batch(batch_size)
			self.session.run([self.update_edit_distance, self.update_loss],
			                 {self.mfcc_lens : mfcc_lens, self.mfccs: mfccs,
			                  self.phone_lens: phone_lens, self.phones: phones})
		return self.session.run([self.current_edit_distance, self.summaries[dataset_name]])[0]

	def predict(self, dataset, batch_size):
		# Done: Predict phoneme sequences for the given dataset.
		phone_id_seqs = []
		while not dataset.epoch_finished():
			mfcc_lens, mfccs, phone_lens, phones = dataset.next_batch(batch_size)
			indices, values = self.session.run(
				[self.predictions[0].indices, self.predictions[0].values],
				{self.mfcc_lens : mfcc_lens, self.mfccs: mfccs,
				 self.phone_lens: phone_lens, self.phones: phones}
			)
			# print("indices:\n{}".format(indices))
			# print("values:\n{}".format(values))

			predictions = [[]] * batch_size
			for index2D, value in zip(indices, values):
				predictions[index2D[0]].append(value)
			# print("predictions:")
			# print(predictions)
			# print("#################################")
			phone_id_seqs.extend(predictions)
		return phone_id_seqs


if __name__ == "__main__":
	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
	parser.add_argument("--rnn_cell_dim", default=5, type=int, help="RNN cell dimension.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	args = parser.parse_args()

	# Create logdir name
	args.logdir = "logs/{}-{}-{}".format(
		os.path.basename(__file__),
		datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
		",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists("logs"): os.mkdir("logs")  # TF 1.6 will do this by itself

	# Load the data
	timit = timit_mfcc26_dataset.TIMIT("timit-mfcc26.pickle")

	# Construct the network
	network = Network(threads=args.threads)
	network.construct(args, len(timit.phones), timit.mfcc_dim)

	# Train
	for i in range(args.epochs):
		network.train_epoch(timit.train, args.batch_size)

		accuracy = network.evaluate("dev", timit.dev, args.batch_size)
		print("{:.2f}".format(100 * accuracy))

	# Predict test data
	with open("{}/speech_recognition_test.txt".format(args.logdir), "w") as test_file:
		# TODO: Predict phonemes for test set using network.predict(timit.test, args.batch_size)
		# and save them to `test_file`. Save the phonemes for each utterance on a single line,
		# separating them by a single space. The phonemes should be printed as strings (use
		# timit.phones to convert phoneme IDs to strings).
		phone_id_seqs = network.predict(timit.test, args.batch_size)
		# print("phone_id_seqs")
		# print(phone_id_seqs)
		for phone_id_seq in phone_id_seqs:
			for phone_id in phone_id_seq:
				print("{} ".format(timit.phones[phone_id]), file=test_file, end='')
			print("", file=test_file)

	for phone_id_seq in phone_id_seqs:
		for phone_id in phone_id_seq:
			print("{} ".format(timit.phones[phone_id]), end='')
		print()
