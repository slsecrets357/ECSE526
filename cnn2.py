import numpy as np
import cv2
import math
import time
from scipy import signal
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward_propagation(self, input):
        raise NotImplementedError
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
class ConvolutionalLayer():
    def __init__(self, kSize):
        pass
    def forward_prop(self):
        pass
    def back_prop(self,output_error,alpha):
        pass
class PoolingLayer():
    def __init__(self, kSize):
        pass
    def forward_prop(self):
        pass
    def back_prop(self,output_error,alpha):
        pass
def correlate_signal(image, kernel):
    return signal.correlate2d(image, kernel, "full")
def convolve_signal(image, kernel):
    return signal.convolve2d(image, kernel, "full")
def processImage(image):
    image = cv2.imread(image) 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
    return image
def max_pool(A, size):
    w = math.ceil(A.shape[0]/2)
    h = math.ceil(A.shape[1]/2)
    out = np.zeros((w,h))
    y=0
    while y<A.shape[1]:
        x=0
        while x<A.shape[0]:
            #print(A[y:y+size,x:x+size])
            out[int(x/size),int(y/size)] = np.amax(A[x:x+size,y:y+size])
            x+=size
        y+=size
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
    image = processImage('image1.jpg')
    print(image.shape)
    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel2 = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]])
#     A = np.arange(25).reshape((5,5))
#     B = max_pool(A, 2)
#     print(A)
#     B = np.pad(A, [(1, 1), (1, 1)], mode='constant', constant_values=0)
#     out1 = convolve(A,kernel,1)
#     out2 = convolve2D(A, kernel, padding=1)
#     print(out1)
#     print(out2)
    # Convolve and Save Output
    t2 = time.time()
    #output = convolve(image, kernel, padding=2)
    output = convolve_signal(image, kernel2)
    #outputMax = max_pool(output, 2)
    print(time.time()-t2)
    cv2.imwrite('correlateSignal2.jpg', output)
    #cv2.imwrite('convolveMaxBird.jpg', outputMax)
