import math

# Sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Tanh
def tanh(x):
    return math.tanh(x)

# ReLU
def relu(x):
    return max(0, x)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

# Softmax (for list)
def softmax(x_list):
    exp_vals = [math.exp(x) for x in x_list]
    total = sum(exp_vals)
    return [val / total for val in exp_vals]


# Testing
x = 2
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))

arr = [2.0, 1.0, 0.1]
print("Softmax:", softmax(arr))