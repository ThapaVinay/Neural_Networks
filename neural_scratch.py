# shape in array gives us the dimension of the array

import numpy as np
X = [2,4,3,2]
np.random.seed(1)  # will get particular random values for a particular seed value

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # returns normal distribution of mean 0 and variance 1
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  
        self.biases = np.zeros((1,n_neurons))

    def forward(self, n_inputs):
        self.outputs = np.dot(n_inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, n_inputs):
        self.outputs = np.maximum(0,n_inputs)        

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,3)

layer1.forward(X)
activation1 = Activation_ReLU()
activation1.forward(layer1.outputs)
print(activation1.outputs)

layer2.forward(activation1.outputs)
activation1.forward(layer2.outputs)
print(activation1.outputs)














'''
inputs = [2,4,3,2]
weights = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]    # 3 neuron output 
biases = [1,2,4]
layer_output = []


import numpy as np
result = (np.dot(weights,inputs) + biases)

print(result)



for weight,bias in zip(weights, biases):   # it iterates two lists together in tuple form
    neuron_output = 0
    for w,i in zip(weight,inputs):
        neuron_output += w*i
    neuron_output += bias
    layer_output.append(neuron_output)

print(layer_output)
'''


