import numpy
import scipy.special

class neuralNetwork():

	input_nodes = 3
	hidden_nodes = 3
	output_nodes = 3

	learning_rate = 0.3
	#n = neuralNetwork(inputNodes, hidden_nodes, output_nodes, learning_rate)

	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		self.inputNodes = inputNodes
		self.hiddenNodes = hiddenNodes
		self.outputNodes = outputNodes

		self.weightsih = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes. self.inputNodes))
		self.weightsho = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

		#learning Rate
		self.learningRate = learningRate

		#activation function is the sigmoid function
		self.activation_function = lambda x:scipy.special.expit(x)


	def train(self, inputs_list, targets_list):
		#convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		target = numpy.array(inputs_list, ndmin=2).T
		hidden_inputs = numpy.dot(self.weightsih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.weightsho, inputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.weightsho.T, output_errors)

		self.weightsho += self.learningRate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))


	def query(self, inputs_list):
		#converts inputs list to 2d array (matrix)
		inputs = numpy.array(inputs_list, ndmin = 2).T

		#calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.weightsih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.weightsho, hidden_outputs)

		#calculate signals from final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs


