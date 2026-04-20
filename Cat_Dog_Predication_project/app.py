import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow Version:", tf.__version__)

# =========================
# 1. Data Preprocessing
# =========================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   # 80% train, 20% validation
)

train_data = datagen.flow_from_directory(
    'dataset/PetImages',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/PetImages',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# =========================
# 2. Model
# =========================

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# =========================
# 3. Compile
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 4. Train
# =========================

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# =========================
# 5. Save Model
# =========================

model.save("cat_dog_model.keras")

print("✅ Model Trained & Saved!")