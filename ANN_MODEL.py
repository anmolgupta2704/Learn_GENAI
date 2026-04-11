import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------
# Ensure df is already loaded (CSV example)
# df = pd.read_csv("your_dataset.csv")


# Dummy dataset
data = {
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "label": [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
# Features (X) and Target (y)
X = df.drop('label', axis=1)
y = df['label']

# ----------------------------------------
# STEP 2: Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# STEP 3: Feature Scaling (IMPORTANT 🔥)
# ----------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# STEP 4: Build ANN Model
# ----------------------------------------
tf.random.set_seed(42)

ann = tf.keras.models.Sequential()

# Input layer + Hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim=X_train.shape[1]))

# Hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Output layer (binary classification)
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ----------------------------------------
# STEP 5: Compile Model
# ----------------------------------------
ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------
# STEP 6: Train Model
# ----------------------------------------
ann.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    verbose=1
)

# ----------------------------------------
# STEP 7: Evaluate Model
# ----------------------------------------
loss, accuracy = ann.evaluate(X_test, y_test)

print(f"\n✅ Test Accuracy: {accuracy:.4f}")