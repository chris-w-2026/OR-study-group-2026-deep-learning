import numpy as np


class NeuralNetwork:
    def __init__(self, layers: list[int]):
        self.layers = layers



if __name__ == "__main__":
    net = NeuralNetwork([3, 5, 5, 2])
    print(net.layers)
