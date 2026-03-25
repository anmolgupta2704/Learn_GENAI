# import math

# # Sigmoid
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

# # Tanh
# def tanh(x):
#     return math.tanh(x)

# # ReLU
# def relu(x):
#     return max(0, x)

# # Leaky ReLU
# def leaky_relu(x, alpha=0.01):
#     return x if x > 0 else alpha * x

# # Softmax (for list)
# def softmax(x_list):
#     exp_vals = [math.exp(x) for x in x_list]
#     total = sum(exp_vals)
#     return [val / total for val in exp_vals]


# # Testing
# x = 2
# print("Sigmoid:", sigmoid(x))
# print("Tanh:", tanh(x))
# print("ReLU:", relu(x))
# print("Leaky ReLU:", leaky_relu(x))

# arr = [2.0, 1.0, 0.1]
# print("Softmax:", softmax(arr))
import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 100)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid()
functions = {
    "Sigmoid": sigmoid(x),
    "Tanh": tanh(x),
    "ReLU": relu(x),
    "Leaky ReLU": leaky_relu(x)
}

for name, y in functions.items():
    plt.figure()
    plt.plot(x, y)
    plt.title(name)
    plt.grid()
    plt.show()
plt.show()