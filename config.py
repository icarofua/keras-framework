# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy
numpy.random.bit_generator = numpy.random._bit_generator
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.optimizers import Adam
from keras.layers import Lambda
import albumentations as albu
from keras_metrics import f1score

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
customs_func = {"f1score": f1score}

POS = 1 #positive class
NEG = 0 #negative clas
batch_size = 128
NUM_EPOCHS = 100
layers = 3
num = 2048
nchannels=3 #number of channels
image_size_w_c = 64 #image´s width for vehicle´s shape
image_size_h_c = 64 #image´s height for vehicle´s shape
tam_max = 4
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
path = '.'
train_augs = [[],[]]

seq_car = albu.Compose(
  [
    albu.IAACropAndPad(px=(0, 8)),
    albu.IAAFliplr(),
    albu.IAAAffine(scale=(0.4, 1.8),rotate=(-3,3),order=[0,1],cval=(0),mode='constant'), #scale 0.8 1.2

  ], p=1.0)

for i in range(tam_max):
  train_augs[0].append(seq_car)
  train_augs[1].append(seq_car)