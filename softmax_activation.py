# input -> exponentiate -> normalise -> output
# exp + normalise = softmax
# normalise = u / $u

import numpy as np
layer_inputs = [[4.8, 1.21 , 2.385],[2.34, 4.3, 4.56]]

# the exponential value of a large input can give your overflow error 
# to remove that we subtract every value with the maximum value  

exp_values = np.exp(layer_inputs)

# axis 1 will sum row-wise  and keepdims keeps the dimensions intact 
norm_values = exp_values / np.sum(exp_values, axis = 1, keepdims= True)

print(norm_values)
