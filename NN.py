import numpy as np

class NN(object):

	def __init__(self, n_input, size_input, activation, n_layers, *n_nodes):
		self.n_layers = n_layers
		self.n_nodes = n_nodes
		self.n_input = n_input
		self.size_input = size_input
		self.activation = self.getActivation(activation)
		self.scale_weight = 0.01

	def set_num_nodes(self, num_layer, num_nodes):
		self.n_nodes[num_layer - 1] = num_nodes

	def create(self):
		self.W = [self.scale_weight * np.random.randn(self.n_nodes[x], self.n_nodes[x+1]) for x in range(self.n_layers - 1)]
		self.W.insert(0, self.scale_weight * np.random.randn(self.size_input, self.n_nodes[0]))
		self.b = [np.zeros((1, x)) for x in self.n_nodes]		

	def getActivation(self, activation):
		return lambda x : np.maximum(0, x)

	def calc_output_gradient(self, output, y):
		exp_output = np.exp(output)
		doutput = exp_output/np.sum(exp_output, axis=1, keepdims=True)
		doutput[range(self.n_input), y] -= 1
		doutput /= self.n_input
		return doutput

	def calc_loss(self, output, y, reg):
		exp_output = np.exp(output)
		probs = exp_output/np.sum(exp_output, axis=1, keepdims=True)
		log_probs = -np.log(probs[range(self.n_input), y])

		data_loss = np.mean(log_probs)
		reg_loss = 0.5 * reg * sum([np.sum(x**2) for x in self.W])

		return data_loss + reg_loss

	def train(self, X, y, learn_rate, reg, epochs):
		outputs = [None] * self.n_layers
		doutput = [None] * self.n_layers
		dW = [None] * self.n_layers
		db = [None] * self.n_layers

		for i in range(epochs):
			data = X
			# forward pass
			for j in range(self.n_layers - 1):
				data = self.activation(np.dot(data, self.W[j]) + self.b[j])
				outputs[j] = data
			# output layer (without activation)
			outputs[-1] = np.dot(data, self.W[-1]) + self.b[-1]

			# calculate loss
			loss = self.calc_loss(outputs[-1], y, reg)
			if i%1000 == 0:
				print i, ' : ', loss

			# calculate gradient of output layer 
			doutput[-1] = self.calc_output_gradient(outputs[-1], y)

			# back propagation
			for j in range(len(self.W) - 1, 0, -1):
				dW[j] = np.dot(outputs[j - 1].T, doutput[j]) + reg * self.W[j]
				db[j] = np.sum(doutput[j], axis=0)
				doutput[j - 1] = np.dot(doutput[j], self.W[j].T)
				doutput[j - 1][outputs[j - 1] <= 0] = 0
			# back propagation to input layer
			dW[0] = np.dot(X.T, doutput[0]) + reg * self.W[0]
			db[0] = np.sum(doutput[0], axis=0)

			# update weights and biases
			for i in range(self.n_layers):
				self.W[i] += -learn_rate * dW[i]
				self.b[i] += -learn_rate * db[i]

	# predict accuracy on test data
	def predict(self, X, y):
		for i in range(self.n_layers - 1):
			X = self.activation(np.dot(X, self.W[i]) + self.b[i])
		scores = np.dot(X, self.W[-1]) + self.b[-1]
		predictions = np.argmax(scores, axis=1)
		print 'accuracy : %.3f' % np.mean(y == predictions)


def generateData():
	N = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in xrange(K):
	  ix = range(N*j,N*(j+1))
	  r = np.linspace(0.0,1,N) # radius
	  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
	  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	  y[ix] = j
	return X,y