import numpy as np


class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        output = self.activation(total)
        return output
