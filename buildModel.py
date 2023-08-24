import numpy as np
import matplotlib.pyplot as plt
import keras
import scipy
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# Define paths for training and test datasets
TRAIN_PATH = "chest-xray-kaggle/train"
TEST_PATH = "chest-xray-kaggle/test"

# Create a Sequential model for a CNN with 4 CNN layers
model = Sequential()

# Add the first convolutional layer with 32 filters, ReLU activation, and input shape of (224, 224, 3)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))

# Add a second convolutional layer with 64 filters and ReLU activation
model.add(Conv2D(64,(3,3), activation='relu'))

# Add a max pooling layer with pool size (2, 2)
model.add(MaxPooling2D(pool_size=(2,2)))

# Add dropout regularization with a rate of 0.25
model.add(Dropout(0.25))

# Add another convolutional layer with 64 filters and ReLU activation
model.add(Conv2D(64,(3,3),activation='relu'))

# Add another max pooling layer with pool size (2, 2)
model.add(MaxPooling2D(pool_size=(2,2)))

# Add dropout regularization with a rate of 0.25
model.add(Dropout(0.25))

# Add another convolutional layer with 128 filters and ReLU activation
model.add(Conv2D(128,(3,3),activation='relu'))

# Add another max pooling layer with pool size (2, 2)
model.add(MaxPooling2D(pool_size=(2,2)))

# Add dropout regularization with a rate of 0.25
model.add(Dropout(0.25))

# Flatten the output for passing through dense layers
model.add(Flatten())

# Add a dense layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))

# Add dropout regularization with a rate of 0.5
model.add(Dropout(0.5))

# Add a final dense layer with 1 unit and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Create an ImageDataGenerator for data augmentation and normalization for training
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create an ImageDataGenerator for normalization for testing
test_datagen = image.ImageDataGenerator(rescale=1./255)

# Create a generator for training data from the specified directory
train_generator = train_datagen.flow_from_directory(
    "chest-xray-kaggle/chest_xray/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

# Create a generator for validation data from the specified directory
validation_generator = test_datagen.flow_from_directory(
    "chest-xray-kaggle/chest_xray/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Train the model using the generator for training data
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)

