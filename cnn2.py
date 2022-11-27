import numpy as np
import cv2
import math
from numpy import newaxis
import time
from scipy import signal

class ConvNet:
    def __init__(self):
        self.layers = []
    def add_layer(self, layer):
        self.layers.append(layer)
    def forward_prop(self, X):
        result = X
        for layer in self.layers:
            result = layer.forward_prop(result)
        return result
    def back_prop(self, loss):
        inputLoss = loss
        for layer in reversed(self.layers):
            pass
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward_prop(self, input):
        raise NotImplementedError
    def back_prop(self, output_error, learning_rate):
        raise NotImplementedError
class FlattenLayer(Layer):
    def __init__(self, inputShape):
        self.inputShape = inputShape
    def forward_prop(self, input):
        return input.reshape((input.shape[0], input.shape[1]+input.shape[2]*2))
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
    print("maxpool A: ", A.shape)
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
if __name__ == '__main__':
    # Grayscale Image
    image = processImage('buggy.jpg', 573)
    image = image[newaxis,:,:]
    image2 = processImage('image1.jpg', 573)
    image2 = image2[newaxis,:,:]
    images = np.array([image,image2])
    print("images: ", images.shape)
    
    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).astype('float')
    kernel2 = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]])
    gaussian = [[0.003, 0.013, 0.022, 0.013, 0.003],[
0.013, 0.059, 0.097, 0.059, 0.013],[
0.022, 0.097, 0.159, 0.097, 0.022],[
0.013, 0.059, 0.097, 0.059, 0.013],[
0.003, 0.013, 0.022, 0.013, 0.003
]]
    # Convolve and Save Output
    l1 = ConvolutionalLayer(images.shape,5,1)
    l1.kernels = np.array(gaussian)
    l1.kernels = l1.kernels[newaxis,newaxis,:,:]
    l1.kernels = np.array([l1.kernels,l1.kernels])
    print("kernels: ", l1.kernels.shape)
    output = l1.forward_prop(images)
    print("output: ", output.shape)
    output22 = output[0,0]
    output02 = output[1,0]
    cv2.imwrite('l1_0.jpg', output22)
    cv2.imwrite('l1_1.jpg', output02)
# 
    l2 = ConvolutionalLayer(output.shape,3,1)
    l2.kernels = np.array(kernel)
    l2.kernels = l2.kernels[newaxis,newaxis,:,:]
    l2.kernels = np.array([l2.kernels,l2.kernels])
    out2 = l2.forward_prop(output)
    print("output2: ", out2.shape)
    output2a = out2[0,0]
    output2b = out2[1,0]
    cv2.imwrite('l2_0.jpg', output2a)
    cv2.imwrite('l2_1.jpg', output2b)
    
    l3 = PoolingLayer(2)
    out3 = l3.forward_prop(out2)
    out3a = out3[0,0]
    out3b = out3[1,0]
    cv2.imwrite('l3_0.jpg', out3a)
    cv2.imwrite('l3_1.jpg', out3b)
    #back3 = l3.back_prop(out3,0.1)
    back3 = out2
    cv2.imwrite('back3_0.jpg', back3[0,0])
    cv2.imwrite('back3_1.jpg', back3[1,0])
    back2 = l2.back_prop(back3,0.1)
    cv2.imwrite('back2_0.jpg', back2[0,0])
    cv2.imwrite('back2_1.jpg', back2[1,0])
    back1 = l1.back_prop(back2, 0.1)
    cv2.imwrite('back1_0.jpg', back1[0,0])
    cv2.imwrite('back1_1.jpg', back1[1,0])