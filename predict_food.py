import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load class names
with open("categories.pkl", "rb") as f:
    class_names = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model("food101_model.h5")

# Dictionary of average calorie values for some foods (modify as needed)
food_calories = {
    "apple_pie": 320, "baby_back_ribs": 570, "baklava": 350, "beef_carpaccio": 120,
    "beef_tartare": 170, "beet_salad": 190, "chicken_wings": 430, "chocolate_cake": 356,
    "donuts": 450, "french_fries": 365, "hamburger": 540, "hot_dog": 290,
    "ice_cream": 207, "omelette": 154, "pizza": 285, "sushi": 200, "waffles": 291
}

def predict_food(image_path):
    """Predicts the food category and estimates calories."""

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])  # Convert to float

    # Estimate calories (default to 250 kcal if unknown)
    estimated_calories = food_calories.get(predicted_label, 250)

    print(f"Predicted Food: {predicted_label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Estimated Calories: {estimated_calories} kcal")

    return predicted_label, confidence, estimated_calories

# Example test
image_path = r"test_food.jpg"
predict_food(image_path)
