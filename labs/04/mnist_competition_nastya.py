#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        with tf.device('/gpu:0'):
            graph = tf.Graph()
            graph.seed = seed
            self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def make_layer_from_specification(self, layer_input, layer_spec):
        layer_spec_parsed = layer_spec.split("-")
        if layer_spec_parsed[0] == "C":
            # construct a conv layer
            [_, nfilters, filter_size, stride, padding] = layer_spec_parsed
            return tf.layers.conv2d(layer_input,
                                    filters=int(nfilters),
                                    kernel_size=int(filter_size),
                                    strides=int(stride),
                                    padding=padding,
                                    activation=tf.nn.relu)
        elif layer_spec_parsed[0] == "CB":
            # construct a conv layer with batch normalization
            [_, nfilters, filter_size, stride, padding] = layer_spec_parsed
            conv_layer = tf.layers.conv2d(layer_input,
                                    filters=int(nfilters),
                                    kernel_size=int(filter_size),
                                    strides=int(stride),
                                    padding=padding,
                                    activation=None)
            conv_layer_bn = tf.layers.batch_normalization(conv_layer, training=self.is_training)
            conv_layer_bn_relu = tf.nn.relu(conv_layer_bn)
            return conv_layer_bn_relu

        elif layer_spec_parsed[0] == "M":
            # construct a max polling layer
            [_, filter_size, stride] = layer_spec_parsed
            return tf.layers.max_pooling2d(layer_input,
                                           pool_size=int(filter_size),
                                           strides=int(stride))

        elif layer_spec_parsed[0] == "F":
            # construct a flatten layer
            return tf.layers.flatten(layer_input)

        elif layer_spec_parsed[0] == "R":
            # construct a dense layer
            [_, hidden_size] = layer_spec_parsed
            return tf.layers.dense(layer_input,
                                   units=int(hidden_size),
                                   activation=tf.nn.relu)

    def construct(self, args, batches_per_epoch, decay_rate):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            features = self.images
            if args.cnn is not None:
                for layer_spec in args.cnn.split(","):
                    features = self.make_layer_from_specification(features, layer_spec)

            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
                global_step = tf.train.create_global_step()
                learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
                                                           batches_per_epoch, decay_rate, staircase=True)
                self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           tf.contrib.summary.scalar("train/learning_rate", learning_rate)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        acc, _, test_labels = self.session.run([self.accuracy, self.summaries[dataset], self.predictions], {self.images: images, self.labels: labels, self.is_training: False})
        return acc, test_labels

    def predict(self, dataset, images):
        return self.session.run(self.predictions, {self.images: images, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate.")

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
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    # set up decay rate
    decay_rate = np.power(args.learning_rate_final / args.learning_rate, 1 / (args.epochs - 1))
    batches_per_epoch = mnist.train.num_examples // args.batch_size

    network = Network(threads=args.threads)
    network.construct(args, batches_per_epoch, decay_rate)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        acc, _ = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        print("Dev: {:.2f}".format(100 * acc))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    test_labels = network.predict("test", mnist.test.images)
    print("Test: {:.2f}".format(100 * acc))
    with open("mnist_competition_test.txt", "w") as test_file:
        for label in test_labels:
            print(label, file=test_file)
