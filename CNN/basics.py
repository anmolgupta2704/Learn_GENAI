import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dataset (built-in dataset)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. Normalize data (0–255 → 0–1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Reshape (CNN expects 3D image)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 4. Build CNN model
model = models.Sequential()

# Convolution layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

# Pooling layer
model.add(layers.MaxPooling2D((2,2)))

# Second Conv layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# Second Pooling
model.add(layers.MaxPooling2D((2,2)))

# Flatten layer
model.add(layers.Flatten())

# Dense layer
model.add(layers.Dense(64, activation='relu'))

# Output layer (10 digits → 10 classes)
model.add(layers.Dense(10, activation='softmax'))

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train model
model.fit(train_images, train_labels, epochs=5)

# 7. Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_acc)