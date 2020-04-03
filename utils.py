import numpy as np
from sklearn import metrics

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

def step_decay(ep):
  if ep < 10:
    lr = 1e-4 * (ep + 1) / 2
  else:
    lr = 1e-4
  print ("lr is ",lr)
  return lr