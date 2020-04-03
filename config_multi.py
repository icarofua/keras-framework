# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import numpy
numpy.random.bit_generator = numpy.random._bit_generator
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.optimizers import Adam
from keras.layers import Lambda
import albumentations as albu
#from imgaug import augmenters as iaa
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

POS = 1 #positive class
NEG = 0 #negative clas
batch_size = 32
NUM_EPOCHS = 100
layers = 3
nchannels=3 #number of channels
image_size_w_c = 128 #image´s width for vehicle´s shape
image_size_h_c = 128 #image´s height for vehicle´s shape
tam_max = 4
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
path = '.'
augs = [[],[]]

seq_car = albu.Compose(
  [
    albu.IAACropAndPad(px=(0, 8)),
    albu.IAAAffine(scale=(0.4, 1.6),order=[0,1],cval=(0),mode='constant'), #scale 0.8 1.2

  ], p=0.7)


for i in range(tam_max):
  augs[0].append(seq_car)
  augs[1].append(seq_car)
