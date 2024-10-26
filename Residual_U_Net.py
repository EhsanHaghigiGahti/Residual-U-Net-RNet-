from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
import os
from keras import models, layers, regularizers
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
#U-Net-32-1024 architecture is used as the base architecture and residual mechanism is added to the network
#The model takes in rgb input and the output is binary classified image.

def Res_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network.
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network

    shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(input)
    shortcut = BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, x])
    res_path = layers.Activation('relu')(res_path)

    return res_path

#Encoder block: Res block followed by maxpooling


def encoder_block(input, num_filters):
    x = Res_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = Res_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_ResNet(input_shape):
    inputs = Input(input_shape)

    s0, p0 = encoder_block(inputs, 32)
    s1, p1 = encoder_block(p0, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = Res_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    d5 = decoder_block(d4, s0, 32)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d5)
    model = Model(inputs, outputs, name="Res-Net")
    return model

seed=24
batch_size= 8
from keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rescale = 1/255.)

mask_data_gen_args = dict(rescale = 1/255.)

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory('#Train Images Path#',
                                                           seed=seed,
                                                           target_size=(512, 512),
                                                           batch_size=batch_size,
                                                           class_mode=None)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory('#Train Masks Path#',
                                                         seed=seed,
                                                         target_size=(512, 512),
                                                         batch_size=batch_size,
                                                         color_mode = 'grayscale',
                                                         class_mode=None)

valid_img_generator = image_data_generator.flow_from_directory('#validation Images Path#',
                                                               seed=seed,
                                                               target_size=(512, 512),
                                                               batch_size=batch_size,
                                                               class_mode=None)
valid_mask_generator = mask_data_generator.flow_from_directory('#validation Masks Path#',
                                                               seed=seed,
                                                               target_size=(512, 512),
                                                               batch_size=batch_size,
                                                               color_mode = 'grayscale',
                                                               class_mode=None)

train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()


from keras import backend as K
def IoU(y_true, y_pred, smooth=100):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (jac) * smooth

def F1_Score(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#def roc_auc(y_true, y_pred):
 #   auc = tf.py_function(roc_auc_score, (y_true, y_pred), tf.float64)
  #  return auc

def overall_accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_samples = K.sum(K.round(K.clip(y_true + y_pred, 0, 1)))
    acc = true_positives / (total_samples + K.epsilon())
    return acc

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]


input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_ResNet(input_shape)

model.compile(optimizer=Adam(learning_rate = 1e-4), loss='binary_crossentropy', metrics=['accuracy',IoU,recall,precision,overall_accuracy])

from keras.callbacks import ModelCheckpoint, CSVLogger
filepath = "#Outputfile path#-{epoch:03d}-{val_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_freq="epoch", mode='min')
log_csv = CSVLogger('#Outputfile path#/LogFile/Model_ResNet_Log.csv', separator=',', append=False);
callbacks_list = [checkpoint, log_csv]


model.summary()

num_train_imgs = len(os.listdir('#Train Images Path#'))
num_val_images = len(os.listdir('#Validation Images Path#'))

train_steps_per_epoch = num_train_imgs //batch_size
val_steps_per_epoch = num_val_images // batch_size

history = model.fit_generator(train_generator, validation_data=val_generator,steps_per_epoch=train_steps_per_epoch,validation_steps=val_steps_per_epoch, epochs=150, callbacks=callbacks_list)
