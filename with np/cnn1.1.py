import copy
import random
import numpy as np
import sys
import math

class Network:
    def __init__(self, layers, biases):
        self.layers = layers
        self.biases = biases
'''
makeConvoluted function basically creates a convoluted layer, i.e. for the intents of this
Neural Network, this type of layer is a list of dictionaries, each dictionary being a neuron.

Each Neuron/dictionary will have a tuple of coordinates as a key which will give the weight for
for that specific input data to be used.

For example, if you represent your data layer in the form of a square within it being a bunch of
squares (each square having some value), then each neuron takes a group of those squares and computes
values.

In this case, we take the top-left corner as (0, 0) and so on.
Note: We only take squares as a value for data, rectangles wouldn't work here.
'''
def makeConvoluted(size, square_size, stride):
    length = int(math.sqrt(size))
    if square_size > length or stride > square_size:
        return None
    #line = [0 for i in range(size)]
    new_layer = []
    for i in range(0, length - square_size + 1, stride):
        for j in range(0, length - square_size + 1, stride):
            #new_line = copy.deepcopy(line)
            new_line = dict()
            for k in range(square_size):
                for l in range(square_size):
                    new_line[(i + k, j + l)] = random.random() * 0.01
            new_layer.append(new_line)
    return new_layer

def makeLayer(no_of_input, no_of_output):
    #x = np.array([[math.random() for j in range(no_of_output)] for i in range(no_of_input)])
    x = np.random.rand(no_of_input, no_of_output) * 0.01
    return x

def form_CNN():
    n = int(input("Enter the number of layers including convoluted ones: "))
    square_size = int(input("Enter the size of square taken in Convoluted layers: "))
    stride = int(input("Enter the stride: "))
    layer_dims = list(input("Enter first the no. of inputs then,\nPress c for each convoluted layer\nOtherwise enter the number of neurons for each layer: ").split())
    network = []
    num = int(input("Now enter the number of colours used to identify the images: "))
    isdone = False
    convoluted = []
    for _ in range(num):
        layers = []
        biases = []
        #convoluted = []
        for i in range(1, len(layer_dims)):
            if str(layer_dims[i]) in "Cc":
                if i == 1:
                    temp = int(layer_dims[0])
                else:
                    temp = len(layers[i - 2])
                    #temp *= temp
                new_layer = makeConvoluted(temp, square_size, stride)
                temp1 = len(new_layer)
                bias = makeLayer(1, temp1)
                layers.append(new_layer)
                biases.append(bias)
                if not isdone:
                    convoluted.append(i - 1)
            else:
                try:
                    layer_dims[i] = int(layer_dims[i])
                    if str(layer_dims[i - 1]) in "Cc":
                        temp1 = len(layers[-1])
                    else:
                        temp1 = layer_dims[i - 1]
                    new_layer = makeLayer(temp1, layer_dims[i])
                    bias = makeLayer(1, layer_dims[i])
                    layers.append(new_layer)
                    biases.append(bias)
                except:
                    print("Error")
                    print(i, layer_dims[i])
                    sys.exit()
        if not isdone:
            isdone = True
        net = Network(layers, biases)
        network.append(net)
    new_layer = []
    for i in range(len(network[-2].layers[-1][0])):
        new_line = dict()
        for j in range(len(network)):
            new_line[j] = random.random() * 0.01
        new_layer.append(new_line)
    new_biases = makeLayer(1, len(new_layer))
    final = Network(new_layer, new_biases)
    network.append(final)
    return network, convoluted

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(y):
    return y * (1 - y)

def lrelu(x, w, b): #w is weights for a layer
    if not isinstance(w, list):
        z = np.dot(x, w) + b
        temp = z.tolist()
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                if temp[i][j] < 0:
                    temp[i][j] *= 0.01
        final = np.array(temp)
        return final
    else:
        b_temp = b.tolist()
        b_temp = b_temp[0]
        if not isinstance(x, list):
            temp = x.tolist()
        else:
            temp = x
        final = [[0 for j in range(len(b_temp))] for i in range(len(temp))]
        length = len(temp[0])
        length = int(math.sqrt(length))
        for i in range(len(w)):
            keys = list(w[i].keys())
            for j in range(len(temp)):
                for k in keys:
                    if isinstance(k, tuple):
                        final[j][i] += temp[j][k[0] * length + k[1]] * w[i].get(k)
                final[j][i] += b_temp[i]
                if final[j][i] < 0:
                    final[j][i] *= 0.01
            if w[i].get(0, -1) == -1:
                w[i][0] = length
        return final

def lrelu_derivative_matrix(y):
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

def forward_propagation(data, network, convoluted):
    Y = [[] for _ in range(len(network) - 1)]
    for j in range(len(network) - 1):
        temp_data = data[j] #In this case, data is a list of matrices
        for i in range(len(network[j].layers)):
            if i < len(network[j].layers) - 1:
                y = lrelu(temp_data, network[j].layers[i], network[j].biases[i])
                Y[j].append(y)
                temp_data = y
            else:
                z = np.dot(temp_data, network[j].layers[i]) + network[j].biases[i]
                y = sigmoid(z)
                Y[j].append(y)
    z = [[0 for _ in range(len(network[-1].layers))] for _ in range(len(Y[0][-1]))]
    #print(network[-1].layers)
    for i in range(len(Y)):
        tempy = Y[i][-1].tolist()
        for j in range(len(tempy)):
            for k in range(len(tempy[j])):
                z[j][k] += tempy[j][k] * network[-1].layers[k][i]
    z = np.array(z)
    z = z + network[-1].biases
    y = sigmoid(z)
    Y.append(y)
    return Y

def backward_propagation(data, network, convoluted, Y, Y_true, learning_rate):
    answer = Y_true
    cw = Y_true.tolist()
    cx = Y[-1].tolist()
    cy = (1 - Y_true).tolist()
    cz = (1 - Y[-1]).tolist()
    for k in range(len(cw)):
        for j in range(len(cw[0])):
            if cx[k][j] == 0:
                cw[k][j] = 0
            else:
                cw[k][j] /= cx[k][j]
            
            if cz[k][j] == 0:
                cy[k][j] = 0
            else:
                cy[k][j] /= cz[k][j]
    cw = np.array(cw)
    cy = np.array(cy)
    main_cost_derivative = cw - cy
    error = main_cost_derivative * sigmoid_derivative(Y[-1])
    errorx = error.tolist()
    temp1 = [sum(errorx[j][k] for j in range(len(errorx))) for k in range(len(errorx[0]))]
    network[-1].biases -= learning_rate * np.array(temp1)
    Mainerror = [np.zeros(np.shape(Y[-1])).tolist() for _ in range(len(network) - 1)]
    for i in range(len(network) - 1):
        tempx = Y[i][-1].tolist()
        for j in range(len(tempx)):
            for k in range(len(tempx[j])):
                Mainerror[i][j][k] += learning_rate * errorx[j][k] * network[-1].layers[k][i]
                network[-1].layers[k][i] += learning_rate * tempx[j][k] * errorx[j][k]
    Mainerror = [np.array(i) for i in Mainerror]
    for main_index in range(len(network) - 1):
        out = True
        for i in range(len(network[main_index].layers) - 1, -1, -1):
            if out:
                '''
                This is standard for a sigmoid function and the assumption is always that
                the weights used here are not in convoluted form as no CNN shows the final
                answer using convoluted layers.
                '''
                error = Mainerror[main_index] * sigmoid_derivative(Y[main_index][i])
                if isinstance(Y[i - 1], list):
                    Y[main_index][i - 1] = np.array(Y[main_index][i - 1])
                adjustment = learning_rate * np.dot(Y[main_index][i - 1].T, error)
                temp = error.tolist()
                temp = [sum([temp[j][i] for j in range(len(temp))]) for i in range(len(temp[0]))]
                temp = np.array(temp)
                error = np.dot(error, network[main_index].layers[i].T)
                bias_adjustment = learning_rate * temp
                network[main_index].layers[i] = network[main_index].layers[i] + adjustment
                network[main_index].biases[i] = network[main_index].biases[i] + bias_adjustment
                out = False
            else:
                if i not in convoluted:
                    '''
                    This is a standard case for Multilayer Perceptrons in general with
                    leaky relu function
                    '''
                    errorx = error * lrelu_derivative_matrix(Y[main_index][i])
                    adjustment = None
                    if i > 0:
                        if isinstance(Y[i - 1], list):
                            Y[main_index][i - 1] = np.array(Y[main_index][i - 1])
                        adjustment = learning_rate * np.dot(Y[main_index][i - 1].T, errorx)
                        if i > 0:
                            error = np.dot(errorx, network[main_index].layers[i].T)
                    else:
                        adjustment = learning_rate * np.dot(data[main_index].T, errorx)
                    temp = errorx.tolist()
                    temp = [sum([temp[j][k] for j in range(len(temp))]) for k in range(len(temp[0]))]
                    temp = np.array(temp)
                    bias_adjustment = learning_rate * temp
                    network[main_index].layers[i] = network[main_index].layers[i] + adjustment
                    network[main_index].biases[i] = network[main_index].biases[i] + bias_adjustment
                else:
                    '''
                    This one deals with the convoluted layer but stays true to the backpropagation
                    formula:
                        adjustment = learning_rate * dC(x, w)/dw
                        ie
                        adjustment = learning_rate * C'(x, w) * activation'(x) * x
                        We essentially keep adding this formula's values to produce a total
                        adjustment.

                        We keep doing this for each and every single weight.
                        However, which weight is used where is confusing and we can't keep
                        changing the value of adjustment for that, hence we directly go to the
                        weight and change the value then and there.
                        
                        Bias adjustment is unaffected because it still uses the matrix form.
                    '''
                    if isinstance(Y[i], list):
                        errorx = error * lrelu_derivative_matrix(np.array(Y[main_index][i]))
                    else:
                        errorx = error * lrelu_derivative_matrix(Y[main_index][i])
                    adjustment = None
                    if not isinstance(network[main_index].layers[i], list):
                        if i > 0:
                            error = np.dot(errorx, network[main_index].layers[i].T)
                        
                        if isinstance(Y[i - 1], list):
                            temp_x = np.array(Y[main_index][i - 1])
                        else:
                            temp_x = Y[main_index][i - 1]
                        adjustment = learning_rate * np.dot(Y[main_index][i - 1].T, errorx)
                        temp = errorx.tolist()
                        temp = [sum([temp[j][k] for j in range(len(temp))]) for k in range(len(temp[0]))]
                        temp = np.array(temp)
                        bias_adjustment = learning_rate * temp
                        network[main_index].layers[i] = network[main_index].layers[i] + adjustment
                        network[main_index].biases[i] = network[main_index].biases[i] + bias_adjustment
                    else:
                        error_temp = errorx.tolist()
                        #error = [[0 for j in range(len(layers[i][0]))] for k in range(len(errorx))]
                        length = network[main_index].layers[i][0].get(0, 0)
                        if i > 0 and length > 0:
                            error = [[0 for j in range(length * length)] for k in range(len(error_temp))]
                            for j in range(len(network[main_index].layers[i])):
                                for k in network[main_index].layers[i][j].keys():
                                    if isinstance(k, tuple):
                                        weight = network[main_index].layers[i][j].get(k)
                                        for l in range(len(error)):
                                            error[l][k[0] * length + k[1]] += weight * error_temp[l][j]
                            error = np.array(error)
                        temp = errorx.tolist()
                        temp = [sum([temp[j][k] for j in range(len(temp))]) for k in range(len(temp[0]))]
                        temp = np.array(temp)
                        bias_adjustment = learning_rate * temp
                        network[main_index].biases[i] = network[main_index].biases[i] + bias_adjustment
                        if i > 0:
                            if isinstance(Y[main_index][i - 1], list):
                                temp_x = Y[main_index][i - 1]
                            else:
                                temp_x = Y[main_index][i - 1].tolist()
                        else:
                            if isinstance(data[main_index], list):
                                temp_x = data[main_index]
                            else:
                                temp_x = data[main_index].tolist()
                        
                        if length > 0:
                            for j in range(len(network[main_index].layers[i])):
                                for k in network[main_index].layers[i][j].keys():
                                    if isinstance(k, tuple):
                                        for l in range(len(error_temp)):
                                            network[main_index].layers[i][j][k] += learning_rate * error_temp[l][j] * temp_x[l][k[0] * length + k[1]]
    return network      
#layers, biases, convoluted = form_CNN()
#print(convoluted)
data = []
'''
for i in range(10):
    dataline = []
    for j in range(81):
        dataline.append(random.random())
    data.append(dataline)
data = [
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
]
'''
data = [
    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
]
data = np.array(data)
network, convoluted = form_CNN()

Y_true = []
for i in range(10):
    yx = [0 for j in range(10)]
    yx[i] = 1
    Y_true.append(yx)
Y_true = np.array(Y_true)
print(Y_true)


for i in range(500):
    Y = forward_propagation(data, network, convoluted)
    #if i % 10 == 0:
     #   print("cost", -(Y_true * np.log(Y[-1]) + (1 - Y_true) * np.log(1 - Y[-1])))
    #layers, biases = backward_propagation(data, layers, biases, convoluted, Y, Y_true, 0.5)
    network = backward_propagation(data, network, convoluted, Y, Y_true, 0.2)
    '''
    Note: It can be assumed that cost_derivative becomes unreliable after learning rate is raised
    to a certain point. Due to that, its seen that the pattern for 1 becomes less clear with increase
    in LR.

    Eg. With 1000 iterations 0.1-0.13 are fine
        With 500 iterations 0.2 is fine
    '''

#data = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
#data = np.array(data)
print(Y[-1])
Y_new = forward_propagation(data, network, convoluted)
print(Y_new[-1])