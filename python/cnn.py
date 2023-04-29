# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_jpkM38g56MLLl4KuCwW11_irICHT0k1
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from __future__ import print_function
import pandas as pd
import shutil
import sys
import random
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

labels = pd.read_csv(r'/content/drive/MyDrive/charts/charts/charts/train_val.csv')

train_dir =r'/content/drive/MyDrive/charts/charts/charts/train_val'
DR = r"/content/drive/MyDrive/charts/charts/charts/train"
if not os.path.exists(DR):
    os.mkdir(DR)
for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(DR +"/"+ str(class_name)):
        os.mkdir(DR +"/"+ str(class_name))
    src_path = train_dir + '/'+ str(filename) + '.png'
    dst_path = DR +"/"+ str(class_name) + '/' + str(filename) + '.png'
    try:
        shutil.copy(src_path, dst_path)
        
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))

data = pd.read_csv(r'/content/drive/MyDrive/charts/charts/charts/train_val.csv')

# set path to parent folder
parent_folder = "/content/drive/MyDrive/charts/charts/charts/train"

# iterate through subfolders and count number of images
for folder in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder)
    if os.path.isdir(folder_path):
        num_images = len([filename for filename in os.listdir(folder_path) if filename.endswith(".png")])
        print("Folder {} contains {} images".format(folder, num_images))

root_dir = '/content/drive/MyDrive/charts/charts/charts/train'
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
val_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Create the train and val directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Define the subfolders
subfolders = ['vbar_categorical', 'hbar_categorical', 'line', 'pie', 'dot_line']

# Create subdirectories in train and val directories
for subfolder in subfolders:
    os.makedirs(os.path.join(train_dir, subfolder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, subfolder), exist_ok=True)

# Split the data into train and val sets
for subfolder in subfolders:
    files = os.listdir(os.path.join(root_dir, subfolder))
    random.shuffle(files)
    train_files = files[:160] # 80% for training
    val_files = files[160:] # 20% for validation
    
    # Copy train files to train directory
    for file_name in train_files:
        src_path = os.path.join(root_dir, subfolder, file_name)
        dst_path = os.path.join(train_dir, subfolder, file_name)
        shutil.copy(src_path, dst_path)
    
    # Copy val files to val directory
    for file_name in val_files:
        src_path = os.path.join(root_dir, subfolder, file_name)
        dst_path = os.path.join(val_dir, subfolder, file_name)
        shutil.copy(src_path, dst_path)

# set path to parent folder
parent_folder = "/content/drive/MyDrive/charts/charts/charts/train_split/train"

# iterate through subfolders and count number of images
for folder in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder)
    if os.path.isdir(folder_path):
        num_images = len([filename for filename in os.listdir(folder_path) if filename.endswith(".png")])
        print("Folder {} contains {} images".format(folder, num_images))

# set path to parent folder
parent_folder = "/content/drive/MyDrive/charts/charts/charts/train_split/val"

# iterate through subfolders and count number of images
for folder in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder)
    if os.path.isdir(folder_path):
        num_images = len([filename for filename in os.listdir(folder_path) if filename.endswith(".png")])
        print("Folder {} contains {} images".format(folder, num_images))

test_dir = '/content/drive/MyDrive/charts/charts/charts/test'

chart_types = {
    0: 'dot_line',
    1: 'hbar_categorical',
    2: 'line',
    3: 'pie',
    4: 'vbar_categorical'

}

def predict_labels(test_dir, model):
    # Get the list of image files in the test directory
    image_files = os.listdir(test_dir)
    num_images = len(image_files)
    
    # Preprocess the images and make predictions on them
    predictions = []
    for i in range(num_images):
        img_path = os.path.join(test_dir, image_files[i])
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch)
        label = np.argmax(prediction)
        print(f"Prediction for {image_files[i]}: {chart_types[label]}")
        # Convert the image to a NumPy array
        img_array = np.array(img)
        # Plot the image using matplotlib
        plt.imshow(img)
        plt.show()
        predictions.append(chart_types[label])
    print(predictions)

# Set the seed for reproducibility
tf.random.set_seed(42)

# Define the data directory paths
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
val_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Define the parameters for the data generator
batch_size = 32
img_height = 224
img_width = 224

# Create the training data generator
train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Create the validation data generator
val_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Generate the training data from the directory
train_data = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Generate the validation data from the directory
val_data = val_data_gen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

plt.plot(history.history['loss'], label='Training Loss',color='green', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss',color='brown', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy', color='green', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='brown', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(val_data)

# Predict the labels for the validation data
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true labels for the validation data
y_true = val_data.classes

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Print the confusion matrix
print("conf_matrix : ","\n")
print(conf_matrix,"\n")
sns.heatmap(conf_matrix, annot=True, cmap="Reds", fmt="d")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

dot_line = '/content/drive/MyDrive/charts/charts/charts/train_split/val/dot_line'
predict_labels(dot_line, model)

hbar_categorical = '/content/drive/MyDrive/charts/charts/charts/train_split/val/hbar_categorical'
predict_labels(hbar_categorical, model)

line = '/content/drive/MyDrive/charts/charts/charts/train_split/val/line'
predict_labels(line, model)

pie = '/content/drive/MyDrive/charts/charts/charts/train_split/val/pie'
predict_labels(pie, model)

vbar_categorical = '/content/drive/MyDrive/charts/charts/charts/train_split/val/vbar_categorical'
predict_labels(vbar_categorical, model)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)

import tensorflow as tf

# Define the input shape
input_shape = (227, 227, 3)

# Define the AlexNet model
model = tf.keras.Sequential([
    # Layer 1
    tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Layer 2
    tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Layer 3
    tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    
    # Layer 4
    tf.keras.layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    
    # Layer 5
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Layer 6
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Layer 7
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Layer 8
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Define the training and validation data generators
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory('/content/drive/MyDrive/charts/charts/charts/train_split/train',
                                                      target_size=input_shape[:2],
                                                      batch_size=32,
                                                      class_mode='categorical')

val_generator = val_data_gen.flow_from_directory('/content/drive/MyDrive/charts/charts/charts/train_split/val',
                                                  target_size=input_shape[:2],
                                                  batch_size=32,
                                                  class_mode='categorical')

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    epochs=20,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // val_generator.batch_size)

# Plot the training and validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0,20)
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print('Validation Accuracy:', val_acc)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)

# Set the image size and batch size
img_size = (224, 224)
batch_size = 32

# Set the train and test data directories
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
test_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Define the data generator for train and test data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the train and test data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')

# Define the VGG16 model
base_model = tf.keras.applications.VGG16(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet')

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add the classification layers on top of the pre-trained model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)
train_loss, train_acc = model.evaluate(train_data)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
print('Train Loss:', train_loss)
print('Train Accuracy:', train_acc)

# Plot the loss and accuracy curves for train and test data
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the validation data
val_loss, val_acc = model.evaluate(val_data)

# Predict the labels for the validation data
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true labels for the validation data
y_true = val_data.classes

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Print the confusion matrix
print("conf_matrix : ","\n")
print(conf_matrix,"\n")
sns.heatmap(conf_matrix, annot=True, cmap="Reds", fmt="d")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Define directories for train and test data
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
test_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Create data generators for train and test data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the VGG19 model
base_model = tf.keras.applications.VGG19(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add the top layers for classification
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, x)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size
)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.3f}')
print(f'Test Accuracy: {test_acc:.3f}')

# Plot the training and validation losses
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)


# Plot the loss and accuracy curves for train and test data
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)

# Define the directories for the train and test data
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
test_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Define the batch size and number of epochs
batch_size = 32
epochs = 20

# Define the image size
img_size = (224, 224)

# Use the ImageDataGenerator class to load the data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the ResNet50 model
base_model = tf.keras.applications.ResNet50(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Fit the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, verbose=2)

# Print the test loss and accuracy
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Plot the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

# Plot the loss and accuracy curves for train and test data
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the image size
img_size = (224, 224)

# Define the batch size
batch_size = 32

# Define the train and test data directories
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
test_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Define the data augmentation parameters for the train data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the data augmentation parameters for the test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Define the train and test data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the GoogleNet model
base_model = tf.keras.applications.InceptionV3(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# Add the classification layers on top of the base model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)
print("Test Loss: {:.3f}\nTest Accuracy: {:.3f}".format(loss, accuracy))

# Plot the loss and accuracy curves for the training and validation data
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.subplot(2, 1, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch')
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)

# Define the directories for the train and test data
train_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/train'
test_dir = '/content/drive/MyDrive/charts/charts/charts/train_split/val'

# Define the batch size and number of epochs
batch_size = 32
epochs = 10

# Define the image size
img_size = (224, 224)

# Use the ImageDataGenerator class to load the data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the Xception model
base_model = tf.keras.applications.Xception(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Fit the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, verbose=2)

# Print the test loss and accuracy
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Plot the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

# Plot the loss and accuracy curves for train and test data
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data)
print('Validation Accuracy:', val_acc)
test_dir = '/content/drive/MyDrive/charts/charts/charts/test'
predict_labels(test_dir, model)