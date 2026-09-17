import numpy as np

from aoife.neurons.single_neuron import output

#we have four inputs/examples. Each of these has two values.
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

#from our four inputs of two values, we expect to get four outputs of one value
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(0)

#2 inputs = 2 hidden neurons
#weight for inputs 1 and 2
W1 = np.random.randn(2, 2)
#one bias for each hidden neuron, initially 0
b1 = np.zeros((1,2))

#output layer weights
# 2 hidden neurons into one output neuron
W2 = np.random.randn(2, 1)
#a single output bias for the output neuron
b2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

# how sensitive is the output? How much should the weight change?
def sigmoid_derivative(x):
    return x*(1-x)

#controls step size
learning_rate = 0.5

#repeat 10,000 times, each pass = one attempt at improving
for epoch in range(10000):

    #forward pass
    hidden = sigmoid(X @ W1 + b1)

    output = sigmoid(hidden @ W2 + b2)

    #error
    error = y - output

    #backprop
    d_output = error * sigmoid_derivative(output)

    d_hidden = (
        d_output @ W2.T
    ) * sigmoid_derivative(hidden)

    #update
    W2 += learning_rate * hidden.T @ d_output
    b2 += learning_rate * np.sum(
        d_output,
        axis=0,
        keepdims=True
    )

    W1 += learning_rate * X.T @ d_hidden
    b1 += learning_rate * np.sum(
        d_hidden,
        axis=0,
        keepdims=True
    )

print(np.round(output, 3))