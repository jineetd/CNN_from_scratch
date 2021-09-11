import numpy as np


def sigmoid(input):
	"""
	Apply the sigmoid function to input.
	input: Input to which the sigmoid activation will apply

	Returns result of sigmoid activation
	"""
	if type(input) in [list, tuple]:
		input = np.array(input)
	return 1.0 / (1 + np.exp(-1 * input))




def relu(input):
	"""
	Apply the relu activation to input.
	input: Input to which the relu activation will apply

	Returns the result of relu activation
	"""
	if not (type(input) in [list, tuple, np.ndarray]):
		if input < 0:
			return 0
		else:
			return input
	elif type(input) in [list, tuple]:
		input = np.array(input)
	result = input
	result[input < 0] = 0
	return result

def softmax(input):
	"""
	Apply softmax activation to input.
	input: Input to which softmax activation will apply.

	Returns the result of softmax activation
	"""
	return input / (np.sum(input) + 0.000001)

