#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# switch off GPU: CUDA_VISIBLE_DEVICES = " " python3


class Network:
	WIDTH = 28
	HEIGHT = 28
	LABELS = 10

	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                               intra_op_parallelism_threads=threads))

	def construct(self, args):
		with self.session.graph.as_default():
			# Inputs
			self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
			self.labels = tf.placeholder(tf.int64, [None], name="labels")
			self.is_training = tf.placeholder(tf.bool, [], name="is_training")

			# Computation
			# Add layers described in the args.cnn. Layers are separated by a comma and can be:
			cnn_desc = args.cnn.split(',')
			depth = len(cnn_desc)
			layers = [None] * (1 + depth)
			layers[0] = self.images
			for l in range(depth):
				layer_idx = l + 1
				layer_name = "layer{}-{}".format(l, cnn_desc[l])
				specs = cnn_desc[l].split('-')
				# print("...adding layer {} with specs {}".format(layer_name, specs))
				if specs[0] == 'C':
					# - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
					#   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
					layers[layer_idx] = tf.layers.conv2d(inputs=layers[layer_idx - 1], filters=int(specs[1]),
					                                     kernel_size=int(specs[2]), strides=int(specs[3]), padding=specs[4],
					                                     activation=tf.nn.relu, name=layer_name)
				if specs[0] == 'M':
					# - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
					layers[layer_idx] = tf.layers.max_pooling2d(inputs=layers[layer_idx - 1], pool_size=int(specs[1]),
					                                            strides=int(specs[2]), name=layer_name)
				if specs[0] == 'F':
					# - F: Flatten inputs
					layers[layer_idx] = tf.layers.flatten(inputs=layers[layer_idx - 1], name=layer_name)
				if specs[0] == 'R':
					# - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
					layers[layer_idx] = tf.layers.dense(inputs=layers[layer_idx - 1], units=int(specs[1]), activation=tf.nn.relu,
					                                    name=layer_name)

			# Store result in `features`.
			features = layers[-1]

			output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
			self.predictions = tf.argmax(output_layer, axis=1)

			# Training
			loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
			global_step = tf.train.create_global_step()
			self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

			# Summaries
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
			summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
			self.summaries = {}
			with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
				self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
				                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
			with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for dataset in ["dev", "test"]:
					self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
					                           tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train(self, images, labels):
		self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels})

	def evaluate(self, dataset, images, labels):
		accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels})
		return accuracy


if __name__ == "__main__":
	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
	parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	args = parser.parse_args()

	# Create logdir name
	args.logdir = "logs/{}-{}-{}".format(
		os.path.basename(__file__),
		datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
		",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

	# Load the data
	from tensorflow.examples.tutorials import mnist
	mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

	# Construct the network
	network = Network(threads=args.threads)
	network.construct(args)

	# Train
	for i in range(args.epochs):
		while mnist.train.epochs_completed == i:
			images, labels = mnist.train.next_batch(args.batch_size)
			network.train(images, labels)

		network.evaluate("dev", mnist.validation.images, mnist.validation.labels)

	accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels)
	print("{:.2f}".format(100 * accuracy))
