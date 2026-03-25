import numpy as np

# -------------------------------
# STEP 1: INPUT DATA
# -------------------------------
# Suppose we have 2 input features
X = np.array([1.0, 2.0])  
# Example: [study_hours, sleep_hours]

# -------------------------------
# STEP 2: WEIGHTS INITIALIZATION
# -------------------------------
# Weights between input layer → hidden layer (2 neurons)
W1 = np.array([[0.5, -0.2],
               [0.3,  0.8]])

# Weights between hidden layer → output layer (1 neuron)
W2 = np.array([[0.7],
               [-1.2]])

# -------------------------------
# STEP 3: BIAS TERMS
# -------------------------------
b1 = np.array([0.1, 0.1])   # bias for hidden layer
b2 = np.array([0.2])        # bias for output layer

# -------------------------------
# STEP 4: ACTIVATION FUNCTION
# -------------------------------
def relu(x):
    return np.maximum(0, x)  # ReLU: max(0, x)

# -------------------------------
# STEP 5: FORWARD PROPAGATION
# -------------------------------

# Hidden layer computation
z1 = np.dot(X, W1) + b1  
# z1 = weighted sum + bias

a1 = relu(z1)  
# Apply activation function

# Output layer computation
z2 = np.dot(a1, W2) + b2  

# Final output (no activation here for simplicity)
output = z2

# -------------------------------
# STEP 6: RESULT
# -------------------------------
print("Input:", X)
print("Hidden Layer Output:", a1)
print("Final Output:", output)