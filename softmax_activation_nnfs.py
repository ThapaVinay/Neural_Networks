from nnfs.datasets import spiral_data
import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # returns normal distribution of mean 0 and variance 1
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0,inputs)    

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True ))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims= True)
        self.outputs = probabilities

X, y = spiral_data(samples=10, classes= 3)

layer1 = Layer_Dense(2,4)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(4, 3)
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.outputs)

layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)

print(activation2.outputs)