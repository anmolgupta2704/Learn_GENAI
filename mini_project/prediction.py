from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = keras.models.load_model("cat_dog_model.h5")

# Load test image
img_path = "test.jpg"   # change this
img = image.load_img(img_path, target_size=(150,150))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🐶 Dog")
else:
    print("🐱 Cat")