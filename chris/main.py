import os
import sys
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppress logging
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.stderr = open(os.devnull, "w")
from tensorflow.keras.datasets import mnist  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore


def load_data() -> tuple[tuple, tuple]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

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
        self.learning_rate = 0.1
        self.num_layers = len(layers)
        self.layers = layers
        self.activation_funcs = [self.activation_func] * (self.num_layers - 2) + [self.output_activation_func]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
    
    def feedforward(self, a):
        activations = [a]
        z_values = []
        for W, b, activation_func in zip(self.weights, self.biases, self.activation_funcs):
            z_values.append(z := W @ a + b)
            activations.append(a := activation_func(z))
        return activations, z_values

    def backprop(self, x, y):
        # C = 1/2 * sum(a^L - y) ** 2
        # z^l = w^l a^{l-1} + b^l
        # a^l = o(z^l)
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        activations, zs = self.feedforward(x)
        # delta^L = dC^L/dz^L = dC^L/da^L * da^L/dz^L = (a^L - y) * o'(z)
        delta = (activations[-1] - y) * self.output_derivative_func(zs[-1])
        # dC^L/db^L = dC^L/dz^L * dz^L/db^L = dC^L/dz^L * 1 = delta^L
        nabla_b[-1] = delta 
        # dC^L/dw^L = dC^L/dw^L * dz^L/dw^L = dC^L/dz^L * a^{l-1} = delta^L * a^{l-1}
        nabla_w[-1] = delta @ activations[-2].T
        for l in range(2, self.num_layers):
            delta = (self.weights[-l + 1].T @ delta) * self.derivative_func(zs[-l])   
            nabla_b[-l] = delta   
            nabla_w[-l] = delta @ activations[-l - 1].T
        # w = w - r * dC/dw, b = b - r * dC/db
        self.weights = [w - self.learning_rate * dw for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, nabla_b)]
    
    def train(self, X, Y, epochs: int):
        for epoch in range(epochs):
            print("Epoch:", epoch + 1)
            for x, y in zip(X, Y):
                self.backprop(x.reshape(-1, 1), y.reshape(-1, 1))
    
    def predict(self, x):
        activations = self.feedforward(x.reshape(-1, 1))[0]
        return np.argmax(activations[-1])
    
    def test_accuracy(self, X, Y):
        count = np.count_nonzero(np.array([self.predict(x) for x in X]) == np.argmax(Y, axis=1))
        return count / len(Y)



if __name__ == "__main__":
    net = NeuralNetwork([784, 16, 16, 10], "sigmoid", "sigmoid")
    data = load_data()
    net.train(*data[0], epochs=2)
    print(f"Accuracy: {100 * net.test_accuracy(*data[1])}%")
