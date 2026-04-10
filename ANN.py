from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
import pandas as pd
def create_ann(X_train, y_train):
    # Define the column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['numerical_feature1', 'numerical_feature2']),
            ('cat', OneHotEncoder(), ['categorical_feature'])
        ]
    )

    # Preprocess the training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Create the ANN model
    model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=200)

    # Fit the model to the training data
    model.fit(X_train_processed, y_train)

    return model, preprocessor
def predict_ann(model, preprocessor, X_test):
    # Preprocess the test data
    X_test_processed = preprocessor.transform(X_test)

    # Make predictions
    predictions = model.predict(X_test_processed)

    return predictions

# Example usage:
# X_train = ...  # Your training data
# y_train = ...  # Your training labels
# X_test = ...   # Your test data
# model, preprocessor = create_ann(X_train, y_train)
# predictions = predict_ann(model, preprocessor, X_test)
x_train = pd.DataFrame({
    'numerical_feature1': [10, 2, 11],
    'numerical_feature2': [5, 8, 3],
    'categorical_feature': ['A', 'B', 'A']
})
y_train = [0, 1, 0]
x_test = pd.DataFrame({
    'numerical_feature1': [15, 3, 12],
    'numerical_feature2': [6, 9, 4],
    'categorical_feature': ['B', 'A', 'B']
})

model, preprocessor = create_ann(x_train, y_train)
print("Model created and trained successfully.")
predictions = predict_ann(model, preprocessor, x_test)
print("Predictions:", predictions)
