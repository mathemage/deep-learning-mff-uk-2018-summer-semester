#!/usr/bin/env python3
import numpy as np


def numpy_entropy(index):
	# Load data distribution, each data point on a line
	data_points = {}
	data_count = 0
	with open("numpy_entropy_data_{}.txt".format(index), "r") as data:
		for line in data:
			line = line.rstrip("\n")
			data_point = line
			if data_point not in data_points.keys():
				data_points[data_point] = 1
			else:
				data_points[data_point] += 1
			data_count += 1
	# normalize
	data_probabilities = {}
	for point, frequency in data_points.items():
		data_probabilities[point] = frequency / data_count

	# Load model distribution, each line `word \t probability`, creating
	# a NumPy array containing the model distribution
	model_probabilities = {}
	with open("numpy_entropy_model_{}.txt".format(index), "r") as model:
		for line in model:
			line = line.rstrip("\n")
			data_point, probability = line.split()
			model_probabilities[data_point] = probability

	words = data_probabilities.keys()
	data_distribution = np.zeros(len(words))
	model_distribution = np.zeros(len(words))
	i = 0
	for w in words:
		data_distribution[i] = data_probabilities[w]
		if w in model_probabilities.keys():
			model_distribution[i] = model_probabilities[w]
		i += 1

	entropy = - np.sum(data_distribution * np.log(data_distribution))
	print("{:.2f}".format(entropy))

	if np.all(model_distribution):   # only non-zero probabilities
		cross_entropy = - np.sum(data_distribution * np.log(model_distribution))
	else:
		cross_entropy = np.inf
	print("{:.2f}".format(cross_entropy))
	kl_div = cross_entropy - entropy  # TODO number - inf = -inf
	print("{:.2f}".format(kl_div))


if __name__ == "__main__":
	for i in range(1, 6):
		print("{}th testcase:".format(i))
		numpy_entropy(index=i)
		print('')
