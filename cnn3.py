import numpy as np
import cv2
import math
from numpy import newaxis
import time
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward_prop(self, input):
        raise NotImplementedError
    def back_prop(self, output_error, learning_rate):
        raise NotImplementedError
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
        # returns output for a given input
    def forward_prop(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
        # computes weights error & bias error wrt output error. returns input error to be back propagated.
    def back_prop(self, output_error, alpha):
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
    def forward_prop(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    # Returns input error of activation function wrt to output error.
    def back_prop(self, output_error, alpha):
        return self.activation_prime(self.input) * output_error
def sigmoid(x):
    return 1 /(1 + np.exp(-x))
def sigmoid_prime(x):
    s = sigmoid(x)
    return s*(1 - s)
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
            output = layer.forward_prop(output)
        result = np.zeros(output.shape[0])
        for i in range(output.shape[0]):
            result[i] = np.argmax(output[i])
        return result
    def predict2(self, input_data):
        input_data = input_data[:,newaxis,:,:,:]
        result = np.zeros(input_data.shape[0])
        for j in range(input_data.shape[0]):
            output = input_data[j]
            for layer in self.layers:
                output = layer.forward_prop(output)
            result[j] = np.argmax(output)
        return result

    def predict_output(self, output):
        result = np.zeros(output.shape[0])
        for i in range(output.shape[0]):
            result[i] = np.argmax(output[i])
        return result
    def sgd(self, x_train, y_train, epochs, alpha): #stochastic gradient descent
        print("using stochastic gradient descent")
        print("x_train shape: ", x_train.shape)
        t1 = time.time()
        x_train = x_train[:,newaxis,:,:,:]
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
                    output = layer.forward_prop(output)

                # compute loss (for display purpose only)
                err += self.loss(oneHotYTrain[j], output)
                
                # backward propagation
                error = self.loss_prime(oneHotYTrain[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_prop(error, alpha)
            
            #evaluate accuracy on test set
            out2 = net.predict2(X_dev)
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f accuracy=%f' % (i+1, epochs, err, accuracy(out2, Y_dev)))
            print("time elapsed: ", time.time()-t1)
            #print('epoch %d/%d   error=%f accuracy=%f' % (i+1, epochs, err, 0))
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
                    output = layer.forward_prop(output)

                # compute loss (for display purpose only)
                err += self.loss(oneHotYTrain[j*batchSize:j*batchSize+batchSize], output)
                
                # backward propagation
                error = self.loss_prime(oneHotYTrain[j*batchSize:j*batchSize+batchSize], output)
                for layer in reversed(self.layers):
                    error = layer.back_prop(error, alpha)

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

class FlattenLayer(Layer):
    def __init__(self, inputShape):
        self.inputShape = inputShape
    def forward_prop(self, input):
        #print("flattening.. input shape: ", input.shape)
        shape2 = input.shape[2]
        #print("new shapes: ", input.shape[0], input.shape[1]*shape2*shape2)
        return input.reshape((input.shape[0], input.shape[1]*shape2*shape2))
    def back_prop(self,output_error, learning_rate):
        return output_error.reshape(self.inputShape)
class ConvolutionalLayer(Layer):
    def __init__(self, inputShape, kernelSize, numKernels):
        numInputs,inputDepth, inputHeight, inputWidth = inputShape
        self.inputShape = inputShape
        self.numInputs = numInputs
        self.depth = numKernels
        self.inputDepth = inputDepth
        self.outputShape = (numInputs, numKernels, inputHeight-kernelSize+1, inputWidth-kernelSize+1)
        self.kernelShape = (numInputs, numKernels, inputDepth, kernelSize,kernelSize)
        self.kernels = np.random.randn(*self.kernelShape)
        self.bias = np.random.randn(*self.outputShape)
    def forward_prop(self, input):
        #print("convolving, input shape: ", input.shape)
        self.input = input
        self.output = np.copy(self.bias)
        for num in range(self.numInputs):
            for i in range(self.depth):
                for j in range(self.inputDepth):
                    #print("num,i,j: ", num,i,j)
                    #print("correlation: ",self.input[num,j].shape,self.kernels[num,i,j].shape)
                    self.output[num,i]+=signal.correlate2d(self.input[num,j], self.kernels[num,i,j],"valid")
#                     test1 = self.output[num,i]
#                     cv2.imwrite('testOut%d.jpg' %(num), test1)
        return self.output
    def back_prop(self,output_error,alpha):
        dKdE = np.zeros(self.kernels.shape)
        dIdE = np.zeros(self.input.shape)
        for num in range(self.numInputs):
            for i in range(self.depth):
                for j in range(self.inputDepth):
                    dKdE[num,i,j] = signal.correlate2d(self.input[num,j],output_error[num,i],"valid")
                    dIdE[num,j]+= signal.convolve2d(output_error[num,i],self.kernels[num,i,j], "full")
        self.kernels -= alpha*dKdE
        self.bias -= alpha*output_error
        return dIdE
class PoolingLayer(Layer):
    def __init__(self, poolSize):
        self.poolSize = poolSize
    def forward_prop(self, A):
        output, self.input = max_pool(A, self.poolSize)
        return output
    def back_prop(self,output_error,alpha):
        numInputs, depthOut, w,h = output_error.shape
        for num in range(numInputs):
            for d in range(depthOut):
                for i in range(w):
                    for j in range(h):
                        mini = self.input[num,d,i*self.poolSize:(i+1)*self.poolSize,j*self.poolSize:(j+1)*self.poolSize]
                        i1, i2 = np.unravel_index(np.argmax(mini, axis=None), mini.shape)
                        self.input[num,d,i1+i*self.poolSize, i2+j*self.poolSize] = output_error[num,d,i,j]
        return self.input

def processImage(image, size):
    image = cv2.imread(image)
    #print(image.shape)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (size,size))
    print("resized: ", image.shape)
    return image
def max_pool(A, size):
    #print("maxpool A: ", A.shape)
    inputA = np.zeros(A.shape)
    w = math.ceil(A.shape[2]/size)
    h = math.ceil(A.shape[3]/size)
    out = np.zeros((A.shape[0],A.shape[1],w,h))
    y=0
    for num in range(A.shape[0]):
        for depth in range(A.shape[1]):
            while y<A.shape[3]:
                x=0
                while x<A.shape[2]:
                    miniA = A[num,depth,x:x+size,y:y+size]
                    out[num,depth,int(x/size),int(y/size)] = np.amax(miniA)
                    i,j = np.unravel_index(np.argmax(miniA, axis=None), miniA.shape)
                    inputA[num,depth,x+i,y+j] = 1
                    x+=size
                y+=size
    return out, inputA
            
def relu(X):
    out = np.copy(X)
    out[out<0] = 0
    return out
def relu_prime(X):
    out = np.copy(X)
    out[out<0] = 0
    out[out>0] = 1
    return out
def convolve(A,k,padding): #actually cross-correlate
    kH = k.shape[0]
    kW = k.shape[1]
    #k = np.flipud(np.fliplr(k)) #turn into convolution
    B = np.pad(A, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)
    w = B.shape[0]-kH+1
    h = B.shape[1]-kW+1
    out = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            #print((sum(B[x:x+kshape[0],y:y+kshape[1]]*k)).shape)
            out[x,y] = np.sum(B[x:x+kH,y:y+kW]*k)
    return out

net = Network()
#for pets
# net.add(ConvolutionalLayer((m,1,48,48), 5, 6))
# net.add(PoolingLayer(2))
# net.add(ConvolutionalLayer((m,6,22,22), 3, 12))
# net.add(PoolingLayer(2))
# net.add(FCLayer(10*10*12, 50))
# net.add(ActivationLayer(sigmoid, sigmoid_prime))
# net.add(FCLayer(50, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))
# net.add(FCLayer(50, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))

#for digits
net.add(ConvolutionalLayer((1,1,28,28), 5, 6))
net.add(PoolingLayer(2))
net.add(ConvolutionalLayer((1,6,12,12), 3, 12))
net.add(FlattenLayer((1,12,10,10)))
net.add(FCLayer(10*10*12, 50))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(50, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(50, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_prime))

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
X_dev.resize((X_dev.shape[0], 1, 28, 28))
print("Xdev shape: ", str(X_dev.shape),", Ydev shape: ", Y_dev.shape)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n].T
X_train = X_train / 255.
X_train.resize((X_train.shape[0], 1, 28, 28))
print("Xtrain shape: ", str(X_train.shape),", Ytrain shape: ", Y_train.shape, ", Xtrain len: ", len(X_train), len(Y_train))

net.sgd(X_train[0:10000], Y_train[0:10000], epochs=75, alpha=0.1)