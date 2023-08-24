import pandas as pd
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# Define paths to the CSV file and image directory
FILE_PATH = "chestxray/metadata.csv"
IMAGES_PATH = "chestxray/images"

# Read the CSV file into a DataFrame
df = pd.read_csv(FILE_PATH)

# Define the target directory for COVID-19 images
TARGET_DIR = "Dataset/Covid"

# Create the target directory if it doesn't exist
if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
    print("Covid folder created")

# Initialize a counter for COVID-19 images
cnt = 0

# Iterate through the rows of the DataFrame
for (i, row) in df.iterrows():
    # Check if the finding is "COVID-19" and the view is "PA" (front view)
    if "COVID-19" in row["finding"] and row["view"] == "PA":
        # Get the filename and paths for the source and target images
        filename = row["filename"]
        image_path = os.path.join(IMAGES_PATH, filename)
        image_copy_path = os.path.join(TARGET_DIR, filename)

        # Copy the image to the target directory
        shutil.copy2(image_path, image_copy_path)

        # Increment the counter
        cnt += 1

# Define paths for sampling images from Kaggle dataset
KAGGLE_FILE_PATH = "chest-xray-kaggle/train/NORMAL"
TARGET_NORMAL_DIR = "Dataset/normal"

# Get a list of image names in the Kaggle directory
image_names = os.listdir(KAGGLE_FILE_PATH)

# Shuffle the list of image names randomly
random.shuffle(image_names)

# Loop to copy a subset of normal images
for i in range(142):
    # Get the image name and paths for source and target images
    image_name = image_names[i]
    image_path = os.path.join(KAGGLE_FILE_PATH, image_name)
    target_path = os.path.join(TARGET_NORMAL_DIR, image_name)

    # Copy the image to the target directory
    shutil.copy2(image_path, target_path)

    # Print progress message
    print("Copying image ", i)



