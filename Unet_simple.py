import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,MaxPooling2D, GlobalMaxPool2D,concatenate, add
from tensorflow.keras.models import Model, load_model

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_en(input_img, n_filters=16, dropout=0.3, batchnorm=True):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    return [c1, c2, c3, c4, c5]
    
    # expansive path
def unet_dec(input_img, layers, n_filters=16, dropout=0.4, batchnorm=True):
    c1, c2, c3, c4, _ = layers
    
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (input_img)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    return c9

def unet_half_en(input_img, n_filters=16, dropout=0.3, batchnorm=True):
    # contracting pat
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
   
    c4 = conv2d_block(p3, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    return [c1, c2, c3, c4]

def unet_half_dec(input_img, n_filters=16, dropout=0.4, batchnorm=True):
    c1,c2,c3,x = input_img
    
    u4 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (x)
    u4 = concatenate([u4, c3])
    u4 = Dropout(dropout)(u4)
    c4 = conv2d_block(u4, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u5 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c2], axis=3)
    u5 = Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    u6 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c1])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    u7 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c6)
    
    return u7
    
def dilated_Conv(en1, en2):
    input_img = add([en1, en2])
    dilate3 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(input_img)
    b9 = BatchNormalization()(dilate3)
    b9 = Dropout(rate=0.2)(b9)
    
    dilate4 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=8, kernel_initializer='he_normal')(b9)
    b10 = BatchNormalization()(dilate4)
    b10 = Dropout(rate=0.2)(b10)
    
    dilate5 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=16, kernel_initializer='he_normal')(b10)
    b11 = BatchNormalization()(dilate5)
    b11 = Dropout(rate=0.2)(b11)
    
    return b11

def reconstruct(dec, dec_half):
    out = concatenate([dec, dec_half])
    out = Conv2D(3, (1, 1), activation='sigmoid')(out)
    return out

def loss(img_true, img_pred):
    alpha = 1
    beta = 0
    gamma = 0.05
    def ssim_loss(y_true, y_pred):
        return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    def mae(y_true, y_pred):
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    def inference_mse_loss(frame_hr, frame_sr):
        content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2+(1e-3)**2))
        return tf.reduce_mean(content_base_loss)

    derain_loss = inference_mse_loss(img_true, img_pred)
    edge_loss = inference_mse_loss(img_true, img_pred)
    train_loss = alpha*derain_loss + beta*edge_loss
    
    return train_loss + gamma*ssim_loss(img_true, img_pred) + alpha*mae(img_true, img_pred)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)