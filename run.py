import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
flag_data = pd.read_csv('flags.csv')

X = []
Y = []

for index, row in flag_data.iterrows():
    img_path = row['image_path']
    label = row['label']

    # Load image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    X.append(img_array)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)


# Split the data into training 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

batch_size = 32
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

# Create a new model and add layers for flag detection
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with data augment
model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=5, validation_data=(x_test, y_test))


defult_image_path = 'input/'
input_image = input("Enter the image file path/name: ")
new_image_path = defult_image_path + input_image

img = cv2.imread(new_image_path)
img = cv2.resize(img, (224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(np.expand_dims(img_array, axis=0))

prediction = model.predict(img_array)

if prediction > 0.5:
    print("\033[92mContains Sri Lankan flag\033[00m")
    print("\033[91mDeveloped by Nadeesha Weerasekara\033[00m")
else:
    print("\033[93mDoes not contain Sri Lankan flag\033[00m")

## All rights reserved