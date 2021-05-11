import os
import numpy as np
import matplotlib.pyplot as plt
import math
plt.style.use("ggplot")

from tqdm.notebook import tqdm
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from Darkproperty import Imgrefinedbydark, Imgrefinedbydarkforhalf

import tensorflow as tf

from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,MaxPooling2D, GlobalMaxPool2D,concatenate, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K
from Unet_simple import unet_en,unet_dec,unet_half_en,unet_half_dec,dilated_Conv,reconstruct,loss,PSNR


# Set some parameters
im_width = 320
im_height = 320
epochs = 30
batch_size=4
path_train = './PATH'
path_test = './PATH'
path_weights ='Dehaze_320x320_35epochs.h5'

# Get and resize train images and masks
def get_data(path,path_test, train=True):
    ids = next(os.walk(path))[2]
    # ids = sorted(ids,key=lambda x: int(os.path.splitext(x)[0]))
    ids = sorted(ids, key=lambda x: float(x.split('_')[0]))
    id_test = next(os.walk(path_test))[2]
    id_test = sorted(id_test,key=lambda x: int(os.path.splitext(x)[0]))
    print(ids,id_test)

    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    X_half = np.zeros((len(ids), im_height//2, im_width//2, 3), dtype=np.float32)

    if train:
        y = np.zeros((len(id_test), im_height, im_width, 3), dtype=np.float32)
        y_half = np.zeros((len(id_test), im_height//2, im_width//2, 3), dtype=np.float32)
    print('Getting and resizing images ... ')
    
    for n, id_ in tqdm(enumerate(ids), total=len(ids), disable=True):
        # Load images
        img = load_img(path + id_, grayscale=False)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 3), mode='constant', preserve_range=True)
        x_img_half = resize(x_img, (im_height//2, im_width//2, 3), mode='constant', preserve_range=True)
        X[n] = x_img/255
        X_half[n] = x_img_half/255

    for nt, id_test in tqdm(enumerate(id_test), total=len(id_test),disable=True):
        img = load_img(path_test + id_test, grayscale=False)
        y_img = img_to_array(img)
        y_img = resize(y_img, (im_height, im_width, 3), mode='constant', preserve_range=True)
        y_img_half = resize(y_img, (im_height//2, im_width//2, 3), mode='constant', preserve_range=True)
        y[nt] = y_img / 255
        y_half[nt] = y_img_half / 255
    # z=np.repeat(y,4,axis=0)
    # z_half=np.repeat(y_half,4,axis=0)
    print('Done!')
    if train:
        return X,X_half, y,y_half
    else:
        return X,X_half


# Get and resize test images and masks
def get_datafortest(path,path_test, train=True):
    ids = next(os.walk(path))[2]
    # ids = sorted(ids,key=lambda x: int(os.path.splitext(x)[0]))
    ids = sorted(ids, key=lambda x: float(x.split('_')[0]))
    id_test = next(os.walk(path_test))[2]
    #id_test = sorted(id_test,key=lambda x: int(os.path.splitext(x)[0]))
    ids = sorted(ids, key=lambda x: float(x.split('_')[0]))
    print(ids,id_test)

    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    X_half = np.zeros((len(ids), im_height//2, im_width//2, 3), dtype=np.float32)

    if train:
        y = np.zeros((len(id_test), im_height, im_width, 3), dtype=np.float32)
        y_half = np.zeros((len(id_test), im_height//2, im_width//2, 3), dtype=np.float32)
    print('Getting and resizing images ... ')
    
    for n, id_ in tqdm(enumerate(ids), total=len(ids), disable=True):
        # Load images
        img = load_img(path + id_, grayscale=False)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 3), mode='constant', preserve_range=True)
        x_img_half = resize(x_img, (im_height//2, im_width//2, 3), mode='constant', preserve_range=True)
        X[n] = x_img/255
        X_half[n] = x_img_half/255

    for nt, id_test in tqdm(enumerate(id_test), total=len(id_test),disable=True):
        img = load_img(path_test + id_test, grayscale=False)
        y_img = img_to_array(img)
        y_img = resize(y_img, (im_height, im_width, 3), mode='constant', preserve_range=True)
        y_img_half = resize(y_img, (im_height//2, im_width//2, 3), mode='constant', preserve_range=True)
        y[nt] = y_img / 255
        y_half[nt] = y_img_half / 255

    print('Done!')
    if train:
        return X,X_half, y,y_half
    else:
        return X,X_half

def Unet_simple():
    input_img_half = Input((im_height//2, im_width//2, 3), name='img_half')
    input_img = Input((im_height, im_width, 3), name='img')

    encoding = unet_en(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    half_encoding = unet_half_en(input_img_half, n_filters=16, dropout=0.05, batchnorm=True)

    bottleneck = dilated_Conv(encoding[-1], half_encoding[-1])

    decoding = unet_dec(bottleneck, encoding, n_filters=16, dropout=0.05, batchnorm=True)
    half_decoding = unet_half_dec(half_encoding, n_filters=16, dropout=0.05, batchnorm=True)

    output = reconstruct(decoding, half_decoding)

    return Model(inputs=[input_img, input_img_half], outputs = [output])

# Call saved model
new_model = Unet_simple()
new_model.compile(optimizer=Adam(), loss=loss, metrics=[PSNR])
new_model.load_weights('Dehaze_Multi_scale_320x320_mse_ssim_mae_35epochs.h5')

#model 평가
path_test = 'C:/Users/jungw/Desktop/새 폴더/새 폴더/Dehaze/O-HAZY/hazy/'
path_test_gt = 'C:/Users/jungw/Desktop/새 폴더/새 폴더/Dehaze/O-HAZY/GT/'

X_test,X_test_half, y_test,y_test_half = get_datafortest(path_test, path_test_gt, train=True)

X_TEST = Imgrefinedbydark(X_test,im_height,im_width)
X_TEST_H = Imgrefinedbydarkforhalf(X_test_half,im_height,im_width)

test_loss, test_acc = new_model.evaluate([X_TEST,X_TEST_H], y_test, verbose=1)
print(test_acc)

preds_val = new_model.predict([X_TEST,X_TEST_H], verbose=1) 

#sample1 test
path_testsam = ('./hazy_test1.jpg','./hazy_test2.jpg',)
plt.figure(figsize=(15,10))
for i in range(len(path_testsam)):
    test_img = load_img(path_testsam[i], grayscale=False)
    test_img = img_to_array(test_img)
    test_img = resize(test_img, (im_width, im_height, 3), mode='constant', preserve_range=True)
    test_img_half = resize(test_img, (im_width//2, im_height//2, 3), mode='constant', preserve_range=True)

    test_img = test_img/255
    test_img_half = test_img_half/255
    X_test = np.expand_dims(test_img, axis=0)
    X_test_half = np.expand_dims(test_img_half, axis=0)

    test = Imgrefinedbydark(X_test,im_height,im_width)
    testh = Imgrefinedbydarkforhalf(X_test_half,im_height,im_width)

    pred_test1 = new_model.predict([test, testh])

    result =  resize(pred_test1[0], (4*im_width, 4*im_height, 3), mode='constant', preserve_range=True)
    test_img =  resize(test_img, (4*im_width, 4*im_height, 3), mode='constant', preserve_range=True)

    plt.subplot(len(path_testsam),2,2*i+1)
    plt.imshow(test_img)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)    
    plt.subplot(len(path_testsam),2,2*i+2)
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
plt.tight_layout()
plt.show()
