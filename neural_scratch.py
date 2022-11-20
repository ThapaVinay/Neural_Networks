inputs = [2,4,3,2]
weights = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
biases = [1,2,4]
layer_output = []

for weight,bias in zip(weights, biases):
    neuron_output = 0
    for w,i in zip(weight,inputs):
        neuron_output += w*i
    neuron_output += bias
    layer_output.append(neuron_output)

print(layer_output)



