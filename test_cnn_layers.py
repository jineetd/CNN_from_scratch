import unittest

from cnn import Input2D

class TestLayers(unittest.TestCase):

	def test_input_layer(self):
		# Test for valid input shape dimensions.
		with self.assertRaises(Exception) as context:
			inp = Input2D((2,))
			print(context.exception)
		# Verify a valid dimension for input layer.
		inp = Input2D((2,3))
		self.assertTupleEqual(inp.input_shape, (2,3,1), msg = "Invalid output returned.")
		self.assertTupleEqual(inp.input_shape, inp.layer_output_size, msg = "Invalid output size layer.")
		# Verify negative dimensions for input layers.
		with self.assertRaises(Exception) as context:
			inp = Input2D((2,-3))
			print(context.exception)


if __name__ == '__main__':
    unittest.main()	

