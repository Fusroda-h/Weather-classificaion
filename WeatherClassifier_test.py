
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

path_test = './PATH'
model_weights = 'Mobilenet_edge_hsv_combined_320x320_55epochs.h5'

test_datagen = ImageDataGenerator(rescale=1./255)

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
          
#TEST
def canny_test(X):
    edges = []
    temp = []
    for img in X:
        edge = cv2.Canny((img*255).astype(np.uint8),threshold_x,threshold_y)
        temp.append(np.sum(edge)/255)
    temp = np.asarray(temp)
    edges.append(temp)
    edges = np.asarray(edges)
    return edges

def test_generator_multiple(test_generator):
    while True:
        Y = test_generator.next()
        Y2 = canny_test(Y[0])
        Y3 = hsv(Y[0])
        yield [Y[0], Y2[0], Y3], Y[1]   #Yield both images and their mutual label

test_generator = test_datagen.flow_from_directory(
    path_test,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False)

# Call saved Model
new_model = Mobilenet_simple()
new_model.compile(loss = "categorical_crossentropy",
                      optimizer ='Adam',
              metrics=['accuracy'])
new_model.load_weights(model_weights)

testgenerator = test_generator_multiple(test_generator)
STEP_SIZE_TEST=test_generator.n//batch_size
total_test = test_generator.n

# Model predict
class_names = ['HAZE', 'RAINY', 'SNOWY', 'SUNNY']
predictions = new_model.predict(testgenerator,steps = STEP_SIZE_TEST, verbose=1)

print('Number of test gen',test_generator.n)
print('prediction shape',predictions.shape)

count = 0
test_images = np.zeros((total_test,IMG_HEIGHT,IMG_WIDTH,3))
test_labels = np.zeros((total_test,4))
for batch, CLS in test_generator:
    for j in range(len(batch)):
        img = batch[j]
        test_images[count]=img
        cls_temp = CLS[j]
        test_labels[count]=cls_temp
        count += 1  
        if count == (total_test):
            break
    if count == (total_test):
        break

test_label_cls=np.argmax(test_labels,axis=1)
prediction_label=np.argmax(predictions,axis=1)
print(prediction_label,prediction_label.shape)
print(test_label_cls,test_label_cls.shape)

# Model evaluate
test_loss, test_acc = new_model.evaluate(testgenerator,steps=STEP_SIZE_TEST, verbose=2)
print(test_acc)

acc_each=np.zeros(4)
for i in range(test_generator.n):
  if prediction_label[i] == test_label_cls[i]:
    if prediction_label[i] == 0:
      acc_each[0] +=1
    elif prediction_label[i] == 1:
      acc_each[1] +=1
    elif prediction_label[i] == 2:
      acc_each[2] +=1
    elif prediction_label[i] == 3:
      acc_each[3] +=1

print(acc_each/100)

# Test image visualization
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label_onehot, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  true_label = np.argmax(true_label_onehot)

  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),  
                                class_names[true_label]),
                                color=color,fontsize=5)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label_onehot = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  true_label = np.argmax(true_label_onehot)
  thisplot = plt.bar(range(4), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 8
num_cols = 5
num_images = num_rows*num_cols
plt.tight_layout
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()