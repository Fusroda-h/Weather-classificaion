
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import tensorflow as tf
from PIL import Image
from matplotlib.pyplot import subplot

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input, ReLU, concatenate, MaxPool2D, GlobalMaxPooling2D
from tensorflow.keras import Model

batch_size = 16
epochs = 10
IMG_HEIGHT = 320
IMG_WIDTH = 320

path_train = './PATH'
path_val = './PATH'
model_weights = 'Mobilenet_edge_hsv_combined_320x320_55epochs.h5'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path_train,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        subset = 'training'
        )

validation_generator = train_datagen.flow_from_directory(
        path_val,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        subset = 'validation'
        )
        
STEP_SIZE_TRAIN = train_generator.n/batch_size
STEP_SIZE_VALID = validation_generator.n/batch_size

def Mobilenet_simple():
  weather_in = Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))
  weather = Conv2D(8, (2, 2),padding='same', activation='relu')(weather_in)
  weather = BatchNormalization()(weather)
  weather = DepthwiseConv2D(8,(2,2),padding='same', activation='relu')(weather)
  weather = Conv2D(16,(1,1))(weather)
  weather = BatchNormalization()(weather)
  weather = DepthwiseConv2D(16,(2,2),padding='same', activation='relu')(weather)
  weather = Conv2D(32,(1,1))(weather)
  weather = BatchNormalization()(weather)
  weather = DepthwiseConv2D(32,(2,2),padding='same', activation='relu')(weather)
  weather = Conv2D(32,(1,1))(weather)
  weather = DepthwiseConv2D(32,(2,2),padding='same', activation='relu')(weather)
  weather = Conv2D(64,(1,1))(weather)
  weather = BatchNormalization()(weather)
  weather = DepthwiseConv2D(64,(2,2),padding='same', activation='relu')(weather)
  weather = Conv2D(64,(1,1))(weather)
  weather = DepthwiseConv2D(64,(2,2),padding='same', activation='relu')(weather)
  weather = Conv2D(128,(1,1))(weather)
  weather = BatchNormalization()(weather)
  weather = MaxPooling2D((2, 2))(weather)
  weather = Flatten()(weather)

  edge_in =  Input(shape=(1,))
  edge = Dense(1, activation = 'relu')(edge_in)

  hsv_in = Input(shape=(1,))
  hsv = Dense(1, activation = 'relu')(hsv_in)

  merged = concatenate([weather, edge, hsv])

  fc = Dense(128, activation = 'relu')(merged)
  drop = Dropout(0.5)(fc)
  fc = Dense(32, activation = 'relu')(drop)
  output = Dense(4, activation = 'softmax')(fc)

  return Model(inputs=[weather_in, edge_in, hsv_in], outputs=[output])

combinedModel = Mobilenet_simple()

# Input data preprocessing
threshold_x = 150
threshold_y = 50

lower_blue = np.array([90,50,50])
upper_blue = np.array([110,255,255])

width = IMG_WIDTH
height = IMG_HEIGHT

def canny(X):
    edges = []
    temp = []
    for img in X:
        edge = cv2.Canny((img*255).astype(np.uint8),threshold_x,threshold_y)
        temp.append(np.sum(edge)/(width))
    temp = np.asarray(temp)
    edges.append(temp)
    edges = np.asarray(edges)
    return edges

def hsv(batch):
    hsv_blue = []
    for img in batch:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        hsv_blue.append(np.sum(mask)/(width*height))
    hsv_blue = np.asarray(hsv_blue)
    return hsv_blue
    
def generate_generator_multiple(train_generator):
    while True:
        X1i = train_generator.next()
        X2i = canny(X1i[0])
        X3i = hsv(X1i[0])
        yield [X1i[0], X2i[0], X3i], X1i[1]  #Yield both images and their mutual label
            
def val_generator_multiple(validation_generator):
    while True:
        X1i = validation_generator.next()
        X2i = canny(X1i[0])
        X3i = hsv(X1i[0])
        yield [X1i[0], X2i[0], X3i], X1i[1]  #Yield both images and their mutual label
            
inputgenerator = generate_generator_multiple(train_generator)
valgenerator = val_generator_multiple(validation_generator)

combinedModel.compile(loss = "categorical_crossentropy",
                      optimizer ='Adam',
              metrics=['accuracy'])

# Model load weights
combinedModel.load_weights(model_weights)

callbacks = [ EarlyStopping(patience=5, verbose=1),
             ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
             ModelCheckpoint('Classify_cpu_6th.h5', verbose=1, save_best_only=True, save_weights_only=True) ]

# Model Train
history = combinedModel.fit(
            inputgenerator,
            steps_per_epoch = STEP_SIZE_TRAIN,
            epochs=epochs,
            validation_data = valgenerator,
            validation_steps = STEP_SIZE_VALID,
            callbacks=callbacks)

combinedModel.save_weights('Mobilenet_edge_hsv_combined_320x320_55epochs.h5')

# Accuracy graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
