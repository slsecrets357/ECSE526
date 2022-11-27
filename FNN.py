import numpy as np
import pandas as pd
from numpy import newaxis
import time
import csv
from sklearn.metrics import accuracy_score
import sklearn
import matplotlib.pyplot as plt
#save hidden parameters of a network into txt file
def save_net(filename, net):
    f = open(filename, 'w')
    editor = csv.writer(f, delimiter= ',')
    f.write(str(len(net.layers)))
    f.write('\n')
    for layer in net.layers:
        if isinstance(layer, ActivationLayer):
            continue
        f.write("layer\n")
        row1 = [layer.weights.shape[0],layer.weights.shape[1],layer.bias.shape[0],layer.bias.shape[1]]
        editor.writerow(row1)
        f.write("weight\n")
        for i in range(layer.weights.shape[0]):
            editor.writerow(layer.weights[i])
        f.write("bias\n")
        for i in range(layer.bias.shape[0]):
            editor.writerow(layer.bias[i])
    f.close()
#create a network by loading hidden parameters from a txt file
def load_net(filename):
    net = Network()
    file = open(filename, 'r')
    data = csv.reader(file, delimiter =',')
    numLayers = int(next(data)[0])
    print("numLayers: ", numLayers)
    for k in range(int(numLayers/2)):
        d2 = next(data)
        d2 = np.array(next(data)).astype(int)
        ws0 = (d2[0])
        ws1 = int(d2[1])
        bs0 = int(d2[2])
        bs1 = int(d2[3])
        weights = np.zeros((ws0, ws1))
        bias = np.zeros((bs0, bs1))
        next(data)
        next(data)
        for i in range(ws0):
            weights[i] = np.array(next(data)).astype(float)
            next(data)
        next(data)
        for i in range(bs0):
            bias[i] = np.array(next(data)).astype(float)
            next(data)
        aLayer = FCLayer(ws0, ws1)
        aLayer.weights = weights
        aLayer.bias = bias
        net.add(aLayer)
        net.add(ActivationLayer(sigmoid, sigmoid_prime))
    return net

# Layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward_propagation(self, input):
        raise NotImplementedError
    def backward_propagation(self, output_error, alpha):
        raise NotImplementedError
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
        # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
        # computes weights error & bias error wrt output error. returns input error to be back propagated.
    def backward_propagation(self, output_error, alpha):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)/input_error.shape[0]
        dBias = np.sum(output_error)/input_error.shape[0]
        # update parameters
        self.weights -= alpha * weights_error
        self.bias -= alpha * dBias
        return input_error
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    # apply activation to input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    # Returns input error of activation function wrt to output error.
    def backward_propagation(self, output_error, alpha):
        return self.activation_prime(self.input) * output_error
    
# activation functions and their derivatives
def tanh(x):
    return np.tanh(x);
def tanh_prime(x):
    return 1-np.tanh(x)**2;
def relu(x):
    return np.maximum(x,0)
def relu_prime(x):
    return x>0
def sigmoid(x):
    return 1 /(1 + np.exp(-x))
def sigmoid_prime(x):
    s = sigmoid(x)
    return s*(1 - s)
def softmax(z):
    expz = np.exp(z)
    return expz/np.sum(expz,axis=1)
def softmax_prime(z):
    return z
#error functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2)); 
def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true); 

class Network:
    def __init__(self):
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime
        np.random.seed(0)

    # append layer
    def add(self, layer):
        self.layers.append(layer)

    # set error function
    def change_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    #takes an m-example input and returns m-result output
    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        result = np.zeros(output.shape[0])
        for i in range(output.shape[0]):
            result[i] = np.argmax(output[i])
        return result
    def predict_output(self, output):
        result = np.zeros(output.shape[0])
        for i in range(output.shape[0]):
            result[i] = np.argmax(output[i])
        return result
    
    # train the network
    def sgd(self, x_train, y_train, epochs, alpha): #stochastic gradient descent
        x_train = x_train[:,newaxis,:]
        # sample dimension first
        samples = len(x_train)
        #one hot encode y train
        oneHotYTrain = one_hot(y_train,10)
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                #print("output: ", output.shape)
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(oneHotYTrain[j], output)
                
                # backward propagation
                error = self.loss_prime(oneHotYTrain[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, alpha)
            
            #evaluate accuracy on test set
            out2 = net.predict(X_dev)
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f accuracy=%f' % (i+1, epochs, err, accuracy(out2, Y_dev)))
            
    def bgd(self, x_train, y_train, epochs, alpha): #batch gradient descent
        # sample dimension first
        print("x_train: ", x_train.shape, ", y_train: ", y_train.shape)
        samples = len(x_train)
        oneHotYTrain = one_hot(y_train,10)
        # training loop
        for i in range(epochs):
            output = x_train
            for layer in self.layers:
                output = layer.forward_propagation(output)
            #err = (self.loss(y_train, output))
            error = self.loss_prime(oneHotYTrain, output)
            acc = accuracy(self.predict_output(output), y_train)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, alpha)
            if i % 50 ==0:
                # calculate average error on all samples
                print('epoch %d/%d   accuracy=%f' % (i+1, epochs, acc))
    
    def mini_bgd(self, x_train, y_train, epochs, alpha, batchSize=25): #mini-batch gradient descent
        # sample dimension first
        samples = len(x_train)
        #number of outputs
        numOutputs = np.amax(y_train)+1
        # training loop
        for i in range(epochs):
            shuffledX, shuffledY = shuffle_batch(x_train, y_train)
            #one hot encode y train
            oneHotYTrain = one_hot(shuffledY,numOutputs)
            err = 0
            for j in range(int(samples/batchSize)):
                # forward propagation
                output = shuffledX[j*batchSize:j*batchSize+batchSize]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(oneHotYTrain[j*batchSize:j*batchSize+batchSize], output)
                
                # backward propagation
                error = self.loss_prime(oneHotYTrain[j*batchSize:j*batchSize+batchSize], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, alpha)

            #evaluate accuracy on test set
            out2 = net.predict(X_dev)
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f accuracy=%f' % (i+1, epochs, err, accuracy(out2, Y_dev)))

#helper function that shuffles training set; used in mini-batch gradient descent
def shuffle_batch(x_train, y_train):
    size = len(y_train)
    assert len(x_train) == size
    permutation = np.random.permutation(size)
    return x_train[permutation], y_train[permutation]
#     y = y_train.reshape((1, y_train.size))
#     batch = np.concatenate([y, x_train.T]).T
#     np.random.shuffle(batch)
#     batch = batch.T
#     shuffledY = batch[0]
#     shuffledX = batch[1:batch.shape[1]].T
#     return shuffledX, shuffledY

#helper function that one hot encodes output 
def one_hot(Y, numOutputs):
    oneHotY = np.zeros((Y.size,numOutputs))
    for i in range(Y.size):
        oneHotY[i, int(Y[i])] = 1
    return oneHotY
#compares prediction to actual labels and computes accuracy
def accuracy(prediction, actual):
    size = prediction.shape[0]
    return np.sum(prediction == actual)/size

#loading training data
data = pd.read_csv('digit-recognizer/train.csv')
data.head()     
data = np.array(data)
global m,n
m,n = data.shape

#separate data into training and validation sets
data_dev = data[0:1000].T
X_dev = data_dev[1:n].T
Y_dev = data_dev[0]
one_hot(Y_dev,10)
X_dev = X_dev / 255.
X_dev = X_dev[:,newaxis,:]
print("Xdev shape: ", str(X_dev.shape),", Ydev shape: ", Y_dev.shape)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n].T
X_train = X_train / 255.
print("Xtrain shape: ", str(X_train.shape),", Ytrain shape: ", Y_train.shape, ", Xtrain len: ", len(X_train), len(Y_train))
print(Y_train.size)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# initialize Network
net = Network()
net.add(FCLayer(28*28, 50))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(50, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(50, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_prime))
#training/loading
t1 = time.time()
#net.bgd(X_train[0:1000], Y_train[0:1000], epochs=4000, alpha=0.1)
net.sgd(X_train[0:1000], Y_train[0:1000], epochs=11, alpha=0.1)
#net.mini_bgd(X_train, Y_train, epochs=10, alpha=0.1)
#save_net('net_minibgd_100_3sigmoid_50.txt', net)
#net = load_net('net_minibgd_100_3sigmoid_50.txt')
print("time taken: ", time.time()-t1)

#testing
data = pd.read_csv('test.csv')
data.head()     
data = np.array(data).T
print("data shape: ", data.shape)
y_test = (data[data.shape[0]-1])
x_test = (data[0:data.shape[0]-1].T)
print("test x,y shapes: ", x_test.shape, y_test.shape)
out2 = net.predict(x_test)
print("accuracy: ", accuracy(out2, y_test))
sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_test,out2)
plt.show()
