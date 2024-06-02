import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('fer2013.csv')

# Preprocessing data
def preprocess(data):
    images = data['pixels'].str.split(" ").tolist()
    images = np.array(images, dtype='float32')
    images = images.reshape((images.shape[0], 48, 48, 1))
    images /= 255.0  # Normalisasi
    labels = tf.keras.utils.to_categorical(data['emotion'], num_classes=7)
    return images, labels

train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']

train_images, train_labels = preprocess(train_data)
val_images, val_labels = preprocess(val_data)

# ImageDataGenerator untuk augmentasi data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
validation_generator = validation_datagen.flow(val_images, val_labels, batch_size=32)
