# File contains the code for different layers of CNN

class Input2D:
	"""
	The input layer for CNN.
	The very first layer in your CNN architechture must be Input Layer
	"""

	def __init__(self, input_shape):
		"""
		input_shape: Shape of the input sample to CNN
		"""

		# Size of input sample cannot have less than 2 dimensions.
		if len(input_shape) < 2:
			raise ValueError("The Input2D class creates input layer for input data with "
				             "alteast 2 dimensions. %d dimensions found!." %len(input_shape))
		# Input sample has 2 dimensions, we set the third dimension to 1.
		input_shape = (input_shape[0], input_shape[1], 1)

		for dim in input_shape:
			if dim <= 0:
				raise ValueError("The dimension size of inputs cannot be less than 0.")
		self.input_shape = input_shape
		# Output size for input layer should be the same as input size.
		self.layer_output_size = input_shape