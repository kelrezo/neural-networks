from NeuralNetwork import *
import numpy as np
from decimal import Decimal
from Trainer import *


def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)
        
        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        #Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        #Return the value we changed to zero:
        perturb[p] = 0
        
    #Return Params to original value:
    N.setParams(paramsInitial)
    return numgrad

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def round(arr):
	return np.around(arr, decimals=3)

def normalize(x):
	return x/np.amax(x, axis=0)

x = np.array(([3,5], [5,1], [10,2], [6,1.5]),dtype=float)
y = np.array(([75], [82], [93], [70]), dtype=float)

x = x/np.amax(x, axis=0)
y = y/100

#Training Data:
trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
trainY = np.array(([75], [82], [93], [70]), dtype=float)

#Testing Data:
testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

#Normalize:
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100 #Max test score is 100

#Normalize by max of training data:
testX = testX/np.amax(trainX, axis=0)
testY = testY/100 #Max test score is 100


net = Neural_Network(Lambda = 0.0001)

T = trainer(net)
T.train(trainX, trainY, testX, testY)

print(net.forward(x))
print(y)

#print(T.optimizationResults)

'''
scalar = 3
cost1 =  net.costFunction(x,y)
x1,y1 = net.W1,net.W2

dJdW1, dJdW2 = net.costFunctionPrime(x,y)
net.W1 = net.W1 + scalar*dJdW1
net.W2 = net.W2 + scalar*dJdW2
cost2 = net.costFunction(x,y)
#print(cost1,cost2)

dJdW1, dJdW2 = net.costFunctionPrime(x,y)
net.W1 = net.W1 - scalar*dJdW1
net.W2 = net.W2 - scalar*dJdW2
cost3 = net.costFunction(x, y)
#print(cost2,cost3)



numgrad = computeNumericalGradient(net, x, y)
grad = net.computeGradients(x,y)
#print(numgrad)
#print(grad)
n1 = np.linalg.norm(numgrad)
n2 = np.linalg.norm(grad)
print(n1/n2)
#print (format_e(Decimal(n1/n2)))
'''
