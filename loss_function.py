import numpy as np

# True values (actual)
y_true = np.array([3.0, -0.5, 2.0, 7.0])

# Predicted values (model output)
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

def mse(y_true, y_pred):
    # Step 1: Find difference (error)
    error = y_true - y_pred
    
    # Step 2: Square the error (penalizes large errors)
    squared_error = error ** 2
    
    # Step 3: Take mean (average)
    return np.mean(squared_error)

print("MSE:", mse(y_true, y_pred))


def mae(y_true, y_pred):
    # Step 1: Absolute difference (ignore sign)
    abs_error = np.abs(y_true - y_pred)
    
    # Step 2: Average
    return np.mean(abs_error)

print("MAE:", mae(y_true, y_pred))


def rmse(y_true, y_pred):
    # Square → Mean → Square root
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

print("RMSE:", rmse(y_true, y_pred))


def binary_cross_entropy(y_true, y_pred):
    # Avoid log(0) error by clipping values
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Apply BCE formula
    loss = -(y_true * np.log(y_pred) + 
             (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)

# Example
y_true_bin = np.array([1, 0, 1])
y_pred_bin = np.array([0.9, 0.2, 0.8])

print("BCE:", binary_cross_entropy(y_true_bin, y_pred_bin))


def categorical_cross_entropy(y_true, y_pred):
    # Clip values for stability
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Apply formula
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    
    return np.mean(loss)

# Example (one-hot encoded)
y_true_cat = np.array([[1, 0, 0],
                       [0, 1, 0]])

y_pred_cat = np.array([[0.8, 0.1, 0.1],
                       [0.2, 0.7, 0.1]])

print("CCE:", categorical_cross_entropy(y_true_cat, y_pred_cat))


def hinge_loss(y_true, y_pred):
    # y_true should be -1 or +1
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Example
y_true_hinge = np.array([1, -1, 1])
y_pred_hinge = np.array([0.8, -0.5, 0.3])

print("Hinge Loss:", hinge_loss(y_true_hinge, y_pred_hinge))


def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))

print("Log-Cosh:", log_cosh_loss(y_true, y_pred))


# SUMMARY:
# | Loss                      | Use                   |
# | ------------------------- | --------------------- |
# | MSE                       | Regression            |
# | MAE                       | Regression            |
# | RMSE                      | Regression            |
# | Binary Cross Entropy      | Binary classification |
# | Categorical Cross Entropy | Multi-class           |
# | Hinge                     | SVM                   |
# | Log-Cosh                  | Smooth regression     |
