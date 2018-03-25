#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.float32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False


class Network:

    LABELS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.windows = tf.placeholder(tf.float32, [None, 2 * args.window + 1], name="windows")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # flattened_input = tf.layers.flatten(self.windows, name="flatten")
            hidden_layer1 = tf.layers.dense(self.windows, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer1")
            hidden_layer2 = tf.layers.dense(hidden_layer1, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer2")
            hidden_layer3 = tf.layers.dense(hidden_layer2, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer3")
            output_layer = tf.layers.dense(hidden_layer3, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            correct_pred = tf.equal(self.predictions, tf.cast(self.labels, tf.int64)) # tf.cast(label_one_hot, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")

            # logs and saves
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            tf.add_to_collection("end_points/observations", self.windows)
            tf.add_to_collection("end_points/actions", self.labels)
            self.saver = tf.train.Saver()
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

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]], {self.windows: windows,
                                                                    self.labels: labels,
                                                                    self.is_training: True})

    def evaluate(self, dataset, windows, labels):
        _, accuracy = self.session.run([self.summaries[dataset], self.accuracy], {self.windows: windows, self.labels: labels, self.is_training: False})
        return accuracy

    def classify(self, windows):
        return self.session.run(self.predictions, {self.windows: windows, self.is_training: False})


    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=100, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=10, type=int, help="Size of the window to use.")
    parser.add_argument("--dropout", default=0.6, type=float, help="Dropout rate")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Dropout rate")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Dropout rate")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train.alphabet)
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        print("Epoch %d:" %(i+1))
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            network.train(windows, labels)
        dev_windows, dev_labels = dev.all_data()

        # evaluate datasets
        accurecy_train = network.evaluate("train", windows, labels)
        accurecy_dev = network.evaluate("dev", dev_windows, dev_labels)
        print("Accuracy train %s vs dev %s" %(str(accurecy_train), str(accurecy_dev)))

    # store the model
    network.save("model/model")
    # Generate uppercased test set
    test_txt, _ = test.all_data()
    test_predictions = network.classify(test_txt)

    with open("tst_result.txt", "w") as text_file:
        for i,char in enumerate(test.text):
            letter_cased = char.upper() if test_predictions[i] else char.lower()
            print(letter_cased, file=text_file, end="")
