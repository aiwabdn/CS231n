import numpy as np
import cPickle as pk
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# constants
delta = 1.
reg = 0.01
W = np.random.rand(3073, 10) * 0.001
train_idx = range(50000)
test_idx = range(50000, 60000)
block_sz = 10000
iters = 1000
stepSize = 0.0001
h = 0.000001
sigmoid = lambda x: 1./(1 + np.exp(-x))
dsigmoid = lambda x: (1. - sigmoid(x)) * sigmoid(x)
tanh = lambda x: 2 * sigmoid(2 * x) - 1

# filenames
fnames = ['data_batch_%i' %i for i in range(1,6)]
fnames.append('test_batch')

# loading data
data = np.zeros((60000, 32*32*3), dtype='uint8')
labels = np.zeros(60000, dtype='uint32')

n = 0
for fname in fnames:
	print fname
	f = open('/home/aiwabdn/data/cifar10/'+fname, 'rb')
	s = pk.load(f)
	data[n:n+block_sz] = s['data']
	labels[n:n+block_sz] = s['labels']
	n = n + block_sz
	f.close()

# display image
def showImg(data):
	img = data.reshape(3, 32, 32).transpose(1, 2, 0)
	plt.imshow(img)
	plt.show()

# Multiclass SVM loss with L2 regularisation
def lossMSVM(x, y, W):
	d = (W.T).dot(x)
	diffs = np.maximum(0, d - d[y, range(len(y))] + delta)
	diffs[y, range(len(y))] = 0
	return np.sum(diffs)/len(y) + reg * (W**2).sum()

# Softmax/Cross-entropy classifier with L2 regularisation
def lossSCE(x, y, W):
	d = (W.T).dot(x)
	d = d - np.max(d, axis=0) # subtract max from each column for numerical stability
	d = np.exp(d) # exponentiation for softmax
	s = np.sum(d, axis=0) # sum columns for normalisation
	sfm = -1 * np.log(np.divide(d[y, range(len(y))], s)) # cross entropy probabilities of the target classes of the data points
	return np.sum(sfm)/len(y) + reg * (W**2).sum()

def randomLocalSearch(lossFunc, x, y, W):
	lossBest = lossFunc(x, y, W)
	for i in range(iters):
		Wtry = W + np.random.rand(3073, 10) * stepSize
		loss = lossFunc(x, y, Wtry)
		if loss < lossBest:
			lossBest = loss
			W = Wtry
			print "%d - %f" % (i + 1, lossBest)
	return W

def test(x, y, W):
	scores = (W.T).dot(x)
	predictions = np.argmax(scores, axis=0)
	return np.mean(predictions == y)

# calculate gradient numerically at point x using centered difference formula
def gradient_numerical(func, x):
	grad = np.zeros(x.shape)

	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  	while not it.finished:
  		ix = it.multi_index
  		print ix
		old_val = x[ix]
		x[ix] = old_val + h
		fxph = func(x)
		x[ix] = old_val - h
		fxmh = func(x)
		grad[ix] = (fxph - fxmh) / (2 * h)
		it.iternext()

	return grad

def gradientMSVM(x, y, W):
	grad = np.zeros(W.shape)

	d = (W.T).dot(x)
	diffs = np.maximum(0, d - d[y] + delta)
	diffs[y] = 0
	f = diffs > 0
	grad = np.array([x,]*W.shape[1]).T
	grad = np.multiply(f, grad)
	grad[:, y] = np.sum(f) * x
	return grad

def gradientMSVMB(x, y, W):
	d = (W.T).dot(x)
	diffs = np.maximum(0, d - d[y, range(len(y))] + delta)
	diffs[y, range(len(y))] = 0
	grad = diffs.dot(x.T)
	return grad.T

def gradientDescent(x, y, W):
	prevLoss = lossMSVM(x, y, W)
	loss = prevLoss - 0.001
	it = 0
	while prevLoss - loss > 0.0001:
		prevLoss = loss
		grad = gradientMSVM(x, y, W)
		W -= 0.00000001 * grad
		loss = lossMSVM(x, y, W)
		it += 1
		print it, ' -> ', loss
	return W

def calculate_gradient(W):
	return lossFunc(Xtrain, Ytrain, W)

# random function forward and backward pass
def randomFuncFBP(x, y):
	sigy = sigmoid(y)
	num = x + sigy
	sigx = sigmoid(x)
	xpy = x + y
	xpysqr = xpy**2
	den = sigx + xpysqr
	invden = 1./den
	out = num * invden
	print out

	dout = 1.
	dinvden = num
	dnum = invden
	
	dden = (-1. / (den**2)) * dinvden
	dxpysqr = 1. * dden
	dxpy = 2. * xpy * dxpysqr

	dsigx = 1. * dden
	dsigy = 1. * dnum

	dx = 1. * dnum
	dx += dsigmoid(x) * dsigx
	dx += 1. * dxpy

	dy = dsigmoid(y) * dsigy
	dy += 1. * dxpy

	return (dx,dy)

def example3layer(input):
	h1 = tanh(np.dot(W1, input) + b1)
	h2 = tanh(np.dot(W2, h1) + b2)
	out = tanh(np.dot(W3, h2) + b3)

	

# Pre-processing training data, centering and adding bias term
x = data.T
x = scale(x)
x = np.vstack((np.ones(len(train_idx) + len(test_idx)), x))

Xtrain = x[:, train_idx]
Ytrain = labels[train_idx]
Xtest = x[:, test_idx]
Ytest = labels[test_idx]
# print test(Xtest, Ytest, W)
# lossFunc = lossSCE
# Wbest = randomLocalSearch(lossFunc, Xtrain, Ytrain, W)
# print test(Xtest, Ytest, Wbest)

# print gradient_numerical(calculate_gradient, W)

# Wbest = gradientDescent(Xtrain, Ytrain, W)
# print test(Xtest, Ytest, Wbest)
x = 3
y = 4
for i in range(50000):
	dx, dy = randomFuncFBP(x, y)
	x = x - dx
	y = y - dy
