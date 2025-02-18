import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Set dataset path
dataset_path = r"C:\OneDrive\Desktop\food-101\food-101"

# Load class names (101 food categories)
with open(os.path.join(dataset_path, "meta/classes.txt"), "r") as f:
    class_names = f.read().splitlines()

# Load training and validation image paths
with open(os.path.join(dataset_path, "meta/train.txt"), "r") as f:
    train_images = f.read().splitlines()

with open(os.path.join(dataset_path, "meta/test.txt"), "r") as f:
    test_images = f.read().splitlines()

# Convert paths to full image file paths and extract labels correctly
train_data, train_labels = [], []
for img in tqdm(train_images[:5000], desc="Loading Training Data"):
    class_name = img.split("/")[0]  # Extract class name correctly
    img_path = os.path.join(dataset_path, "images", img + ".jpg")
    
    train_data.append(load_img(img_path, target_size=(224, 224)))
    train_labels.append(class_names.index(class_name))

test_data, test_labels = [], []
for img in tqdm(test_images[:1000], desc="Loading Test Data"):
    class_name = img.split("/")[0]  # Extract class name correctly
    img_path = os.path.join(dataset_path, "images", img + ".jpg")
    
    test_data.append(load_img(img_path, target_size=(224, 224)))
    test_labels.append(class_names.index(class_name))

# Convert images to numpy arrays and normalize
X_train = np.array([img_to_array(img) / 255.0 for img in train_data])
y_train = tf.keras.utils.to_categorical(np.array(train_labels), num_classes=101)

X_test = np.array([img_to_array(img) / 255.0 for img in test_data])
y_test = tf.keras.utils.to_categorical(np.array(test_labels), num_classes=101)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained weights

# Build the model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(101, activation="softmax")(x)  # 101 food categories

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Save the model
model.save("food101_model.h5")
print("Model saved as 'food101_model.h5'")
