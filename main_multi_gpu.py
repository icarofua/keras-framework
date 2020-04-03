from keras.optimizers import Adam
from config import *
import json
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import *
from sys import argv
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import multi_gpu_model

class MultiGPUCheckpoint(ModelCheckpoint):
    
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model


#------------------------------------------------------------------------------
def def_model(model, input1, num, layers):
  input_C = Input(input1)
  convnet_car = model(input1)
  encoded_l_C = convnet(input_C)
  for i in range(layers-1):
    x = Dense(num, activation='relu')(x)
    x = Dropout(0.5)(x)
  x = Dense(num, kernel_initializer='normal',activation='relu')(x)
  x = Dropout(0.5)(x)
  predF2 = Dense(2,kernel_initializer='normal',activation='softmax', name='class_output')(x)
  optimizer = Adam()

  model = multi_gpu_model(Model(inputs=input_C, outputs=predF2), gpus=2)
  model.compile(loss='binary_crossentropy', 
                optimizer=optimizer, 
                metrics=['acc',
                         f1score])

  return model

if __name__ == '__main__':
  num_classes = 10
  epochs = 12
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = np_utils.to_categorical(y_train, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)
  num_train, num_test = x_train.shape[0], x_test.shape[0]  
  f1 = 'model.h5'
  input1 = (image_size_h_c,image_size_w_c,nchannels)
  model = def_model(smallvgg, input1, num, layers)
  lrate = LearningRateScheduler(step_decay)
  es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
  mc = MultiGPUCheckpoint(f1, monitor='val_acc', mode='max', save_best_only=True)
  callbacks_list = [lrate, es, mc]

  #fit model
  model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))