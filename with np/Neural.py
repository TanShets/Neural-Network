import numpy as np
import random
import copy
def sigmoid(M):
    return (1 / (1 + np.exp(-M)))

def derivative(Y_x):
    return Y_x * (1 - Y_x)

n = int(input("Enter the number of inputs: "))
n1 = int(input("Enter the number of factors: "))
X = []
Y = []
W = []
for i in range(n):
    row = list(map(float, input("Enter the values with the output in the end: ").strip().split()))
    X.append(row[:-1])
    Y.append([row[-1]])
X = np.array(X)
Y = np.array(Y)

W = [[random.random() for j in range(n1)] for i in range(7)]
W = np.array(W)
W = W.T

B1 = [random.random() for i in range(7)]
B1 = np.array(B1)
B1 = B1.T

W2 = [[random.random()] for i in range(7)]
W2 = np.array(W2)

B2 = [random.random()]
B2 = np.array(B2)
learning_rate = 0.8
error_rate = 0.4
for steps in range(10000):
    temp = np.dot(X, W) + B1
    temp1 = sigmoid(temp)
    y_temp = np.dot(temp1, W2) + B2
    Y_temp = sigmoid(y_temp)
    Main_Error = Y - Y_temp
    adjustment = learning_rate * np.dot(temp1.T, Main_Error * derivative(Y_temp))
    prod1 = (Main_Error * derivative(Y_temp)).tolist()
    B2 = B2 + learning_rate * np.array([sum([prod1[i][j] for i in range(len(prod1))]) for j in range(len(prod1[0]))])
    error_sum = Main_Error * derivative(Y_temp)
    errors = error_rate * np.dot(error_sum, W2.T)
    W2 = W2 + adjustment
    adjustment = learning_rate * np.dot(X.T, errors * derivative(temp1))
    W = W + adjustment
    prod2 = (errors * derivative(temp1)).tolist()
    B1 = B1 + np.array([sum([prod2[i][j] for i in range(len(prod2))]) for j in range(len(prod2[0]))])
print(Y_temp.tolist())

'''
This is neural network with:
n : 7 : 1 neurons
'''