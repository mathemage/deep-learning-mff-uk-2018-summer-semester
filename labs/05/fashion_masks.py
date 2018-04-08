#!/usr/bin/env python3
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
import numpy as np
import tensorflow as tf


class Dataset:
	def __init__(self, filename, shuffle_batches=True):
		data = np.load(filename)
		self._images = data["images"]
		self._labels = data["labels"] if "labels" in data else None
		self._masks = data["masks"] if "masks" in data else None

		self._shuffle_batches = shuffle_batches
		self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(len(self._images))

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def masks(self):
		return self._masks

	def next_batch(self, batch_size):
		batch_size = min(batch_size, len(self._permutation))
		batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
		return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None, self._masks[
			batch_perm] if self._masks is not None else None

	def epoch_finished(self):
		if len(self._permutation) == 0:
			self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(
				len(self._images))
			return True
		return False


class Network:
	WIDTH = 28
	HEIGHT = 28
	LABELS = 10

	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                             intra_op_parallelism_threads=threads))

	def make_layer_from_specification(self, layer_input, layer_spec):
		layer_spec_parsed = layer_spec.split("-")
		if layer_spec_parsed[0] == "C":
			# construct a conv layer
			[_, nfilters, filter_size, stride, padding] = layer_spec_parsed
			return [tf.layers.conv2d(layer_input,
			                         filters=int(nfilters),
			                         kernel_size=int(filter_size),
			                         strides=int(stride),
			                         padding=padding,
			                         activation=tf.nn.relu)]

		elif layer_spec_parsed[0] == "CB":
			# construct a conv layer with batch normalization
			[_, nfilters, filter_size, stride, padding] = layer_spec_parsed
			conv_layer = tf.layers.conv2d(layer_input,
			                              filters=int(nfilters),
			                              kernel_size=int(filter_size),
			                              strides=int(stride),
			                              padding=padding,
			                              activation=None,
			                              use_bias=False)
			conv_layer_bn = tf.layers.batch_normalization(conv_layer, training=self.is_training)
			conv_layer_bn_relu = tf.nn.relu(conv_layer_bn)
			return [conv_layer, conv_layer_bn, conv_layer_bn_relu]

		# elif layer_spec_parsed[0] == "DB":
		#     # construct a deconv layer with batch normalization
		#     [_, nfilters, filter_size, stride, padding] = layer_spec_parsed
		# 	deconv_layer = tf.layers.conv2d_transpose(layer_input,
		# 																						filters=int(nfilters),
		# 																						kernel_size=int(filter_size),
		# 																						strides=int(stride),
		# 																						padding=padding,
		# 																						activation=None,
		# 																						use_bias=False)
		# 	deconv_layer_bn = tf.layers.batch_normalization(deconv_layer, training=self.is_training)
		# 	deconv_layer_bn_relu = tf.nn.relu(deconv_layer_bn)
		# 	return [deconv_layer, deconv_layer_bn, deconv_layer_bn_relu]

		elif layer_spec_parsed[0] == "M":
			# construct a max-pooling layer
			[_, filter_size, stride] = layer_spec_parsed
			return [tf.layers.max_pooling2d(layer_input,
			                                pool_size=int(filter_size),
			                                strides=int(stride))]
		elif layer_spec_parsed[0] == "D":
			# construct a dropout layer
			[_, dropout_rate] = layer_spec_parsed
			return [tf.layers.dropout(layer_input,
			                          rate=float(dropout_rate),
			                          training=self.is_training)]

		elif layer_spec_parsed[0] == "F":
			# construct a flatten layer
			return [tf.layers.flatten(layer_input)]

		elif layer_spec_parsed[0] == "R":
			# construct a dense layer
			[_, hidden_size] = layer_spec_parsed
			return [tf.layers.dense(layer_input,
			                        units=int(hidden_size),
			                        activation=tf.nn.relu)]

	def construct(self, args, batches_per_epoch, decay_rate):
		with self.session.graph.as_default():
			# Inputs
			self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
			self.labels = tf.placeholder(tf.int64, [None], name="labels")
			self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
			self.is_training = tf.placeholder(tf.bool, [], name="is_training")

			# Computation and training
			layers = [self.images]
			if args.cnn is not None:
				for layer_spec in args.cnn.split(","):
					layers += self.make_layer_from_specification(layers[-1], layer_spec)

			output_layer_labels = tf.layers.dense(layers[-1], self.LABELS, activation=None, name="output_layer_prediction")
			output_layer_mask = tf.layers.dense(layers[-1], self.HEIGHT * self.WIDTH * 2, activation=None,
			                                    name="output_layer_mask")
			output_layer_mask_resh = tf.reshape(output_layer_mask, shape=[-1, self.HEIGHT, self.WIDTH, 1, 2])

			# - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
			self.labels_predictions = tf.argmax(output_layer_labels, axis=1)
			self.masks_predictions = tf.to_float(tf.argmax(output_layer_mask_resh, axis=4))

			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
			only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
			                              self.masks_predictions, tf.zeros_like(self.masks_predictions))
			intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1, 2, 3])
			self.iou = tf.reduce_mean(
					intersection / (tf.reduce_sum(only_correct_masks, axis=[1, 2, 3])
					                + tf.reduce_sum(self.masks, axis=[1, 2, 3]) - intersection)
			)

			# - loss is stored in `loss`
			# loss = loss_mask + loss_pred
			# loss = loss_mask * loss_pred
			loss_pred = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=output_layer_labels, scope="loss")
			loss_mask = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.masks, tf.int64),
			                                                   logits=output_layer_mask_resh, scope="loss")
			# loss_mask = tf.losses.mean_squared_error(labels=tf.cast(self.masks, tf.int64), predictions=output_layer_mask_resh,
			#                                          scope="loss")
			# label_agreement = loss_pred
			# correct_masks = label_agreement * self.masks_predictions
			# intersection_for_loss = tf.reduce_sum(correct_masks * self.masks, axis=[1, 2, 3])
			# loss = tf.reduce_mean(
			# 		intersection_for_loss / (tf.reduce_sum(correct_masks, axis=[1, 2, 3])
			# 		                + tf.reduce_sum(self.masks, axis=[1, 2, 3]) - intersection_for_loss)
			# )
			# correct_label_similarities = tf.one_hot(self.labels, depth=self.LABELS) - tf.nn.softmax(logits=output_layer_labels)
			# correct_masks = (1 - tf.reduce_sum(correct_label_similarities * correct_label_similarities)) * self.masks_predictions
			# intersection_for_loss = tf.reduce_sum(correct_masks * self.masks, axis=[1, 2, 3])
			# loss = tf.reduce_mean(
			# 		intersection_for_loss / (tf.reduce_sum(correct_masks, axis=[1, 2, 3])
			# 		                         + tf.reduce_sum(self.masks, axis=[1, 2, 3]) - intersection_for_loss)
			# )

			# loss_class = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=output_layer_labels, scope="loss")
			# class_cosine_distance = tf.reduce_sum(tf.one_hot(self.labels, depth=self.LABELS) * tf.nn.softmax(output_layer_labels))
			# loss_intersection = tf.reduce_sum(self.masks_predictions * self.masks, axis=[1, 2, 3])
			# loss_iou = tf.reduce_mean(
			# 		loss_intersection / (tf.reduce_sum(self.masks_predictions, axis=[1, 2, 3])
			# 		                + tf.reduce_sum(self.masks, axis=[1, 2, 3]) - loss_intersection)
			# )
			# loss = loss_class * loss_iou
			# loss = class_cosine_distance * loss_iou
			# loss = loss_pred - loss_pred * loss_mask
			# loss = loss_pred + (1 / loss_pred) * loss_mask
			# loss = loss_pred + (0.01 / loss_pred) * loss_mask
			# loss = loss_pred + (0.5 / loss_pred) * loss_mask
			# loss = loss_pred + (0.75 / loss_pred) * loss_mask
			# loss = loss_pred + (0.7 / loss_pred) * loss_mask
			# loss = loss_pred + (0.65 / loss_pred) * loss_mask
			loss = loss_pred + (0.7 / loss_pred) * loss_mask

			global_step = tf.train.create_global_step()
			learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
			                                           batches_per_epoch, decay_rate, staircase=True)
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(extra_update_ops):
				# - training is stored in `self.training`
				self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

			# Summaries
			summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
			self.summaries = {}
			with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
				self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
				                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
				                           tf.contrib.summary.scalar("train/iou", self.iou),
				                           tf.contrib.summary.image("train/images", self.images),
				                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
			with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for dataset in ["dev", "test"]:
					self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
					                           tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy),
					                           tf.contrib.summary.scalar(dataset + "/iou", self.iou),
					                           tf.contrib.summary.image(dataset + "/images", self.images),
					                           tf.contrib.summary.image(dataset + "/masks", self.masks_predictions)]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train(self, images, labels, masks):
		self.session.run([self.training, self.summaries["train"]],
		                 {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

	def evaluate(self, dataset, images, labels, masks):
		[_, iou] = self.session.run([self.summaries[dataset], self.iou],
		                            {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})
		return iou

	def predict(self, images):
		return self.session.run([self.labels_predictions, self.masks_predictions],
		                        {self.images: images, self.is_training: False})


if __name__ == "__main__":
	print("tf.VERSION == {}".format(tf.VERSION))

	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	parser.add_argument("--cnn", default="CB-10-3-2-same,M-3-2,F,R-100", type=str,
	                    help="Description of the CNN architecture.")
	parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
	parser.add_argument("--learning_rate_final", default=0.005, type=float, help="Final learning rate.")
	args = parser.parse_args()

	# Create logdir name
	args.logdir = "logs/{}-{}-{}".format(
			os.path.basename(__file__),
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
			          for key, value in sorted(vars(args).items()))).replace("/", "-")
	)
	if not os.path.exists("logs"): os.mkdir("logs")  # TF 1.6 will do this by itself

	# Load the data
	train = Dataset("fashion-masks-train.npz")
	dev = Dataset("fashion-masks-dev.npz")
	test = Dataset("fashion-masks-test.npz", shuffle_batches=False)

	# Construct the network
	decay_rate = np.power(args.learning_rate_final / args.learning_rate, 1 / (args.epochs - 1))
	batches_per_epoch = len(train._images) // args.batch_size

	network = Network(threads=args.threads)
	network.construct(args, batches_per_epoch, decay_rate)

	# Train
	for i in range(args.epochs):
		j = 0
		while not train.epoch_finished():
			print("Epoch #{} \t Batch #{}".format(i, j))
			j += 1
			images, labels, masks = train.next_batch(args.batch_size)
			network.train(images, labels, masks)

		iou = network.evaluate("dev", dev.images, dev.labels, dev.masks)
		print("Dev: {:.2f}".format(100 * iou))

	# Predict test data
	with open("{}/fashion_masks_test.txt".format(args.logdir), "w") as test_file:
		while not test.epoch_finished():
			images, _, _ = test.next_batch(args.batch_size)
			labels, masks = network.predict(images)
			for i in range(len(labels)):
				print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
