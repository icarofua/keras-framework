from keras.optimizers import Adam
import numpy as np
from config_multi import *
import json
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import *
from sys import argv
import os
from sequence_class import SiameseSequence
from sklearn import metrics
from keras.utils import multi_gpu_model

#------------------------------------------------------------------------------
def calculate_metrics(ytrue1, ypred1):
    conf = metrics.confusion_matrix(ytrue1, ypred1, [0,1])
    maxres = (conf[1,1],
              conf[0,0],
              conf[0,1],
              conf[1,0],
        metrics.precision_score(ytrue1, ypred1) * 100,
        metrics.recall_score(ytrue1, ypred1) * 100,
        metrics.f1_score(ytrue1, ypred1) * 100,
        metrics.accuracy_score(ytrue1, ypred1) * 100)
    return maxres

#------------------------------------------------------------------------------
def test_report(model_name, model, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("%s_inferences_output.txt" % (model_name), "w")
    ytrue, ypred = [], []
    for i in range(len(test_gen)):
      X, Y, paths = test_gen[i]
      Y_ = model.predict(X)
      for y1, y2, p0, p1 in zip(Y_.tolist(), Y.argmax(axis=-1).tolist(), paths[0], paths[1]):
        y1_class = np.argmax(y1)
        ypred.append(y1_class)
        ytrue.append(y2)
        a.write("%s;%s;%d;%d;%s\n" % (p0, p1, y2, y1_class, str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()

'''
def step_decay(ep):
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
'''

def step_decay(ep):
  if ep < 10:
    lr = 1e-4 * (ep + 1) / 2
  else:
    lr = 1e-4
  print ("lr is ",lr)
  return lr
#------------------------------------------------------------------------------
def siamese_model(model, input1, num, layers):
  left_input_C = Input(input1)
  right_input_C = Input(input1)
  convnet_car = model(input1)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)
  inputs = [left_input_C, right_input_C]

  # Add the distance function to the network
  x = L1_layer([encoded_l_C, encoded_r_C])
  for i in range(layers-1):
    x = Dense(num, activation='relu')(x)
    x = Dropout(0.5)(x)
  x = Dense(num, kernel_initializer='normal',activation='relu')(x)
  x = Dropout(0.5)(x)
  predF2 = Dense(2,kernel_initializer='normal',activation='softmax', name='class_output')(x)
  optimizer = Adam()

  model = multi_gpu_model(Model(inputs=inputs, outputs=predF2), gpus=2)
  model.compile(loss='binary_crossentropy', #[binary_focal_loss()], #'binary_crossentropy', 
                optimizer=optimizer, 
                metrics=['acc',
                         f1score])

  return model
#------------------------------------------------------------------------------
def small_vgg_car(input_shape):
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

if __name__ == '__main__':

  tam = int(argv[1])
  # data = json.load(open('dataset%d_1.json' % (tam)))
  data = json.load(open('dataset_1.json'))
  np.random.seed(32)
  trn_0 = []
  trn_1 = []
  for d in data['train']:
    if d[2]==0:
      trn_0.append(d)
    else:
      trn_1.append(d)

  trn_0 = np.random.permutation(trn_0)[:tam].tolist()
  trn_1 = np.random.permutation(trn_1)[:tam].tolist()
  trn = trn_0 + trn_1
  val = data['validation']
  tst = data['test']
  f1 = 'model.h5'
  input1 = (image_size_h_c,image_size_w_c,nchannels)
  siamese_net = siamese_model(small_vgg_car, input1, num, layers)
  lrate = LearningRateScheduler(step_decay)
  es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
  mc = MultiGPUCheckpoint(f1, monitor='val_acc', mode='max', save_best_only=True)
  callbacks_list = [lrate, es, mc]

  trnGen = SiameseSequence(trn, train_augs)
  valGen = SiameseSequence(val)
  valGen2 = SiameseSequence(val, with_paths=True)
  tstGen = SiameseSequence(tst, with_paths=True)

  #fit model
  history = siamese_net.fit_generator(trnGen,
                                  epochs=NUM_EPOCHS,
                                  validation_data=valGen,
                                  callbacks=callbacks_list)

  siamese_net = load_model(f1, custom_objects=customs_func)
  test_report('validation_shape',siamese_net, valGen2)
  test_report('test_shape',siamese_net, tstGen)
