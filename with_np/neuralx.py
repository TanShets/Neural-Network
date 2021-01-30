import numpy as np
import math
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(y):
    return y * (1 - y)

def lrelu(x, w, b):
    z = np.dot(x, w) + b
    temp = z.tolist()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if temp[i][j] < 0:
                temp[i][j] *= 0.01
    final = np.array(temp)
    return final

def lrelu_derivative(y):
    y1 = y.tolist()
    y_derivative = []
    for i in range(len(y1)):
        y_row = []
        for j in range(len(y1[i])):
            if y1[i][j] >= 0:
                y_row.append(1)
            else:
                y_row.append(0.01)
        y_derivative.append(y_row)
    y_derivative = np.array(y_derivative)
    return y_derivative

def forward_propagation(data, layers, biases):
    Y = []
    temp = data
    for i in range(len(layers)):
        if i < len(layers) - 1:
            y = lrelu(temp, layers[i], biases[i])
            Y.append(y)
            temp = y
        else:
            z = np.dot(temp, layers[i]) + biases[i]
            y = sigmoid(z)
            Y.append(y)
    return Y

def cost(Y_true, Y):
    diff = Y_true - Y
    diff *= diff
    diff = diff.tolist()
    diff = [sum(diff[j][i] for j in range(len(diff))) for i in range(len(diff[0]))]
    diff = np.array(diff)
    return diff

def backward_propagation(data, layers, biases, Y, Y_true, learning_rate, isprinty):
    answer = Y_true
    out = True
    error = None
    '''
    if isprinty:
        print(cost(Y_true, Y[-1]))
    '''
    for i in range(len(layers) - 1, -1, -1):
        if out:
            #print(answer * np.log(Y[i]) + (1 - answer)*np.log(1 - Y[i]))
            #cost_derivative = (answer / Y[i]) + (1 - answer) / (1 - Y[i]) #We're using cost function Ylogy + (1 - Y)log(1 - y)
            error = (Y_true - Y[i]) * sigmoid_derivative(Y[i])
            adjustment = learning_rate * np.dot(Y[i - 1].T, error)
            temp = error.tolist()
            temp = [sum([temp[j][i] for j in range(len(temp))]) for i in range(len(temp[0]))]
            temp = np.array(temp)
            error = np.dot(error, layers[i].T)
            bias_adjustment = learning_rate * temp
            layers[i] = layers[i] + adjustment
            biases[i] = biases[i] + bias_adjustment
            out = False
        else:
            errorx = error * lrelu_derivative(Y[i])
            adjustment = None
            if i > 0:
                adjustment = learning_rate * np.dot(Y[i - 1].T, errorx)
                error = np.dot(errorx, layers[i].T)
            else:
                adjustment = learning_rate * np.dot(data.T, errorx)
            temp = errorx.tolist()
            temp = [sum([temp[j][i] for j in range(len(temp))]) for i in range(len(temp[0]))]
            temp = np.array(temp)
            bias_adjustment = learning_rate * temp
            layers[i] = layers[i] + adjustment
            biases[i] = biases[i] + bias_adjustment
    return layers, biases

def makeLayer(no_of_input, no_of_output):
    #x = np.array([[math.random() for j in range(no_of_output)] for i in range(no_of_input)])
    x = np.random.rand(no_of_input, no_of_output) * 0.01
    return x

no_of_layers = None
layers_num = None
no_of_layers = int(input("Enter the number of layers: "))
layers_num = list(map(int, input("Enter the number of inputs first and then the number of neurons in each layer after that: ").strip().split()))
layers = []
biases = []
for i in range(1, len(layers_num)):
    layer = makeLayer(layers_num[i - 1], layers_num[i])
    bias = makeLayer(1, layers_num[i])
    layers.append(layer)
    biases.append(bias)
for i in range(len(layers)):
    print("Weight", np.shape(layers[i]), " Bias", np.shape(biases[i]))
m = int(input("Enter the number of lines of data: "))
data = []
Y_true = []
for i in range(m):
    dataline = list(map(float, input("Enter data with the last number being the output: ").strip().split()))
    data.append(dataline[:-1])
    Y_true.append([dataline[-1]])
data = np.array(data)
Y_true = np.array(Y_true)
print(layers)
learning_rate = float(input("Assign a learning rate for the Neural Network: "))
for i in range(10000):
    Y = forward_propagation(data, layers, biases)
    if i % 1000 == 0:
        print("Value: ", Y[-1])
        isprinty = True
    else:
        isprinty = False
    #print(Y[-1])
    layers, biases = backward_propagation(data, layers, biases, Y, Y_true, learning_rate, isprinty)
Y = forward_propagation(data, layers, biases)
print()
print("Values:\n\n", Y[-1])
print()
print(layers)
print(Y[len(Y) - 1])