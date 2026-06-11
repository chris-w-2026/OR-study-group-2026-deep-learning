import random
import numpy as np

def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return max(0, z)


activation_func_pairs = {"sigmoid": (sigmoid, sigmoid_prime)}


class NeuralNetwork:
    def __init__(self, layers: list[int], activation_func_name: str, output_activation_func_name: str):
        self.activation_func, self.derivative_func = activation_func_pairs[activation_func_name]
        self.output_activation_func, self.output_derivative_func = activation_func_pairs[output_activation_func_name]
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.z_values = []
        self.activations = []

    def feedforward(self, a):
        for b, W in zip(self.biases, self.weights):
            z = W @ a + b
            self.z_values.append(z)
            a = self.activation_func(z)
            self.activations.append(a)
        return a
    
    def backprop(self):
        ...
    
    def train(self):
        ...

    @staticmethod
    def cost(output_activations, y):
        return (output_activations - y) ** 2

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y



if __name__ == "__main__":
    net = NeuralNetwork([3, 5, 5, 2], "sigmoid", "sigmoid")
