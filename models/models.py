from sklearn import metrics
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model,Sequential
from keras import optimizers
from keras.regularizers import l2
from keras.applications import *
from keras.callbacks import *
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config import *
from math import ceil
import json
from sklearn import metrics
from collections import Counter
from keras import backend as K
from keras.layers import *
from keras.models import Model, load_model
import string
import pandas as pd
from imgaug import augmenters as iaa
from keras.optimizers import Adam
import efficientnet.keras as efn 
from attention_module import *
from resnet_v1 import resnet_v1
from resnet_v2 import resnet_v2
from losses import *
# global constants
DIM_ORDERING = 'tf'
CONCAT_AXIS = -1
def inception_module(x, params, dim_ordering, concat_axis,
                     subsample=(1, 1), activation='relu',
                     border_mode='same', weight_decay=None):

    # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
    # file-googlenet_neon-py

    (branch1, branch2, branch3, branch4) = params

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    pathway1 = Convolution2D(branch1[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)

    pathway2 = Convolution2D(branch2[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)
    pathway2 = Convolution2D(branch2[1], 3, 3,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway2)

    pathway3 = Convolution2D(branch3[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(x)
    pathway3 = Convolution2D(branch3[1], 5, 5,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway3)

    pathway4 = MaxPooling2D(pool_size=(1, 1), dim_ordering=DIM_ORDERING)(x)
    pathway4 = Convolution2D(branch4[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             dim_ordering=dim_ordering)(pathway4)

    return Concatenate()([pathway1, pathway2, pathway3, pathway4])


def conv_layer(x, nb_filter, nb_row, nb_col, dim_ordering,
               subsample=(1, 1), activation='relu',
               border_mode='same', weight_decay=None, padding=None):

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=dim_ordering)(x)

    if padding:
        for i in range(padding):
            x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def GoogLeNet(input_shape):
    img_input = Input(input_shape)
    x = conv_layer(img_input, nb_col=7, nb_filter=64,
                   nb_row=7, dim_ordering=DIM_ORDERING, padding=3)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)

    x = conv_layer(x, nb_col=1, nb_filter=64,
                   nb_row=1, dim_ordering=DIM_ORDERING)
    x = conv_layer(x, nb_col=3, nb_filter=192,
                   nb_row=3, dim_ordering=DIM_ORDERING, padding=1)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)

    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    # AUX 1 - Branch HERE
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    # AUX 2 - Branch HERE
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = AveragePooling2D(strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Flatten()(x)

    return Model(img_input,x)



def matchnet(input_shape):
    input1 = Input(input_shape) 
    x = Conv2D(24, kernel_size=(7, 7),activation='relu')(input1)

    # pool0
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # conv1
    x = Conv2D(64, kernel_size=(5, 5),
                activation='relu')(x)
    # pool1
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # conv2
    x = Conv2D(96, kernel_size=(3, 3),
                activation='relu')(x)
    # conv3
    x = Conv2D(96, kernel_size=(3, 3),
                activation='relu')(x)
    # conv4
    x = Conv2D(64, kernel_size=(3, 3),
                activation='relu')(x)
    # pool4
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    # bottleneck
    x = Flatten()(x)
    return Model(input1,x)

def lenet5(input_shape):
    input1 = Input(input_shape) 
    x = Conv2D(20, [5, 5], padding='same', activation='relu')(input1)

    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Conv2D(50, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Flatten()(x)
    return Model(input1,x)

def mccnn(input_shape):
    input1 = Input(input_shape) 
    x = Conv2D(112, (3,3), activation='relu')(input1)
    x = Conv2D(112, (3,3), activation='relu' )(x)
    x = Conv2D(112, (3,3), activation='relu' )(x)
    x = Conv2D(112, (3,3), activation='relu' )(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    return Model(input1,x)

def resnet6(input_shape):

    input1 = Input(input_shape) 

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    x = Flatten()(x5)

    return Model(input1,x)

def resnet8(input_shape):

    input1 = Input(input_shape)
    
    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)

    return Model(input1,x) 

def efficientnet_model(input_shape):
  model1 = efn.EfficientNetB7(weights='imagenet')
  for layer in model1.layers:
    layer.trainable = False
  return Model(inputs=model1.input,outputs=model1.get_layer("avg_pool").output)

def resnet50_model(input_shape):
  model1 = resnet50.ResNet50()
  for layer in model1.layers:
    layer.trainable = False
  return Model(inputs=model1.input,outputs=model1.get_layer("avg_pool").output)

def resnet50_cbam_v1(input_shape):
  return resnet_v1(input_shape, 50, attention_module='cbam_block')

def resnet50_cbam_v2(input_shape):
  return resnet_v2(input_shape, 56, attention_module='cbam_block')

def resnet50_v1(input_shape):
  return resnet_v1(input_shape, 50, attention_module=None)

def resnet50_v2(input_shape):
  return resnet_v2(input_shape, 56, attention_module=None)

def vgg16_model(input_shape):
  model1 = vgg16.VGG16()
  for layer in model1.layers:
    layer.trainable = False
  return Model(inputs=model1.input,outputs=model1.get_layer("flatten").output)


def vgg_original (input_shape):
    input1 = Input(input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    return Model(input1,x)

def vgg_cnn_m_1024(input_shape=(224,224,3)):
    input1 = Input(input_shape)

    x = Conv2D(96, (7, 7), strides=2, activation='relu', padding='same')(input1)
    x = BatchNormalization()(x)   
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(256, (5, 5), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)   
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #x = attach_attention_module(x, 'cbam_block')
    x = Flatten()(x)
    #x = Dense(4096, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(1000, activation='softmax')(x)

    return Model(input1,x)

def small_vgg(input_shape):
    kernels = [64, 128, 128, 256 , 512]
    num_convs = [1,1,1,1,1]
    with_bn = True
    with_l2 = False
    input1 = Input(input_shape)
    x = input1
    c = 0
    for k in kernels:
      for n in range(num_convs[c]):
        if with_l2:
          x = Conv2D(k, 
		(3, 3), 
		kernel_initializer='he_normal', 
		kernel_regularizer=l2(1e-3), 
		padding='same')(x)
        else:
          x = Conv2D(k, 
		(3, 3), 
		kernel_initializer='he_normal', 
		padding='same')(x)
        if with_bn:
          x = BatchNormalization(epsilon=1.001e-5)(x)
          x = Activation('relu')(x)
      c+=1
      x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)

    return Model(input1,x)

def small_vgg3d(input_shape):
    kernels = [64, 128, 128, 256 , 512]
    num_convs = [1,1,1,1,1]
    with_bn = True
    with_l2 = False
    input1 = Input(input_shape)
    x = input1
    c = 0
    for k in kernels:
      for n in range(num_convs[c]):
        if with_l2:
          x = Conv3D(k, 
		(1, 3, 3), 
		kernel_initializer='he_normal', 
		kernel_regularizer=l2(1e-3), 
		padding='same')(x)
        else:
          x = Conv3D(k, 
		(1, 3, 3), 
		kernel_initializer='he_normal', 
		padding='same')(x)
        if with_bn:
          x = BatchNormalization(epsilon=1.001e-5)(x)
          x = Activation('relu')(x)
      c+=1
      x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2))(x)

    x = Flatten()(x)

    return Model(input1,x)

def step_decay2(ep):
  if ep < 10:
    lr = 1e-4 * (ep + 1) / 2
  elif ep < 40:
    lr = 1e-3
  elif ep < 70:
    lr = 1e-4 
  elif ep < 100:
    lr = 1e-5 
  elif ep < 130:
    lr = 1e-6
  elif ep < 160:
    lr = 1e-4 
  else:
    lr = 1e-5 

  print ("lr is ",lr)
  return lr

def step_decay(ep):
  if ep < 10:
    lr = 1e-4 * (ep + 1) / 2
  else:
    lr = 1e-4
  print ("lr is ",lr)
  return lr

class MultiGPUCheckpoint(ModelCheckpoint):
    
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model
