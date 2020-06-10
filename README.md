# Neural-Network
Neural.py has a standard neural network of n inputs and 2 layers with:
Layer 1: 7 Neurons
Layer 2: 1 Neuron

neuralx.py allows you set up certain hyperparameters to form your Neural Network

cnn1.0.py is a cnn with adjustable hyperparameters for a single colour, i.e., convolution exists only for one layer

cnn1.1.py deals with adjustable hyperparameters as well as multiple colours:
Essentially, this CNN will have parallel networks computing the pattern of the image based on the number of colours put.
Then in the end, these results are superposed onto each other, i.e. the result from the first neuron from the first colour pattern is also added up (linearly added w/ weights) with the result of the first neuron of the result of the second colour pattern until the nth colour.

They combine together to form the final answer.
However, there are limitations to this model in case of a 5 x 5 pixel input with one convolution and 10 possible outputs:
1. At 1000 iterations beyond l.r. 0.13, the second result, i.e. the pattern for the number 1 gives an incorrect answer.
    Possible reason: The cost_derivative function might reach its computational limit and give the value as 0 for that answer.
    Thus at that point, the answer can suddenly switch from 1 to 0.
2. (Same reason) At 10000, 0.01 seems suitable.
3. At 500, 0.2 is feasible.
