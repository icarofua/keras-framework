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


def get_transforms(data):   
    if data == 'train':
        return Compose([
            IAACropAndPad(px=(0, 8)),
            IAAFliplr(),
            IAAAffine(scale=(0.4, 1.8),rotate=(-3,3),order=[0,1],cval=(0),mode='constant'),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])