#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
	# Load data distribution, each data point on a line
	data_points = {}
	data_count = 0
	with open("numpy_entropy_data.txt", "r") as data:
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
	with open("numpy_entropy_model.txt", "r") as model:
		for line in model:
			line = line.rstrip("\n")
			data_point, probability = line.split()
			model_probabilities[data_point] = probability

	words = set(data_probabilities.keys()).union(model_probabilities.keys())
	data_distribution = np.zeros(len(words))
	model_distribution = np.zeros(len(words))
	i = 0
	for w in words:
		if w in data_probabilities.keys():
			data_distribution[i] = data_probabilities[w]
		if w in model_probabilities.keys():
			model_distribution[i] = model_probabilities[w]
		i += 1

	if np.all(data_distribution):   # only non-zero probabilities
		entropy = - np.sum(data_distribution * np.log(data_distribution))
	else:
		entropy = np.inf
	print(entropy)
	print("{:.2f}".format(entropy))

	if np.all(model_distribution):   # only non-zero probabilities
		cross_entropy = - np.sum(data_distribution * np.log(model_distribution))
	else:
		cross_entropy = np.inf
	print("{:.2f}".format(cross_entropy))
	kl_div = cross_entropy - entropy  # TODO inf - inf = nan
	print("{:.2f}".format(kl_div))

if __name__ == '__main__':
	print("\t data_probabilities")
	for point, frequency in data_points.items():
		print("{} {} {}".format(point, frequency, data_probabilities[point]))
	print("\t model_probabilities")
	for point, frequency in model_probabilities.items():
		print("{} {}".format(point, frequency))

	print("\t words")
	for w in words:
		print(w)

	print("\t data_distribution")
	print(data_distribution)
	print("\t model_distribution")
	print(model_distribution)
	print("\t entropy")
	print("{:.2f}".format(entropy))
	print("\t cross_entropy")
	print("{:.2f}".format(cross_entropy))
	print("\t kl_div")
	print("{:.2f}".format(kl_div))
