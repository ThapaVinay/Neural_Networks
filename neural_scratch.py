# shape in array gives us the dimension of the array

inputs = [2,4,3,2]
weights = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
biases = [1,2,4]
layer_output = []

import numpy as np
result = (np.dot(weights,inputs) + biases)

print(result)


'''
for weight,bias in zip(weights, biases):   # it iterates two lists together in tuple form
    neuron_output = 0
    for w,i in zip(weight,inputs):
        neuron_output += w*i
    neuron_output += bias
    layer_output.append(neuron_output)

print(layer_output)
'''


