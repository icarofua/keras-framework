from tensorflow.python.keras.utils.data_utils import Sequence
from keras.preprocessing import image
import numpy as np
from keras.utils import np_utils
from config import batch_size, image_size_h_c, image_size_w_c, nchannels, num_workers
#------------------------------------------------------------------------------
def process_load(f1, vec_size):
    _i1 = image.load_img(f1, target_size=vec_size)
    _i1 = image.img_to_array(_i1, dtype='uint8')
    _i1 = ((_i1/255.0) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return _i1

def load_img(img, vec_size, vec_size2, metadata_dict, preprocess):
  iplt0 = process_load(img[0][0], vec_size)
  iplt1 = process_load(img[1][0], vec_size)

  d1 = { "i0":iplt0,
        "i1":iplt1,
        "l":img[2],
        "p1":img[0][0],
        "p2":img[1][0],
        "c1":img[3][0],
        "c2":img[4][0]
        }

  return d1


class SiameseSequence(Sequence):
    def __init__(self,
                features, 
                augmentations=None,
                batch_size=batch_size,
                preprocess = None,
                input2=(image_size_h_c,image_size_w_c,nchannels), 
                with_paths=False):
        self.features = features
        self.batch_size = batch_size
        self.vec_size = input2
        self.preprocess = preprocess
        self.augment = augmentations
        self.with_paths = with_paths

    def __len__(self):
        return int(np.ceil(len(self.features) / 
            float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = self.features[start:end]
        futures = []
        _vec_size = (len(batch),) + self.vec_size
        b1 = np.zeros(_vec_size)
        b2 = np.zeros(_vec_size)
        blabels = np.zeros((len(batch)))
        p1 = []
        p2 = []
        c1 = []
        c2 = []

        i1 = 0
        for _b in batch:
            res = load_img(_b, self.vec_size, self.vec_size2, self.metadata_dict, self.preprocess)
            if self.augment is not None:
                b1[i1,:,:,:] = self.augment[0][0](image=res['i0'])["image"]
                b2[i1,:,:,:] = self.augment[1][0](image=res['i1'])["image"]
            else:
                b1[i1,:,:,:] = res['i0']
                b2[i1,:,:,:] = res['i1']
            blabels[i1] = res['l']
            p1.append(res['p1'])
            p2.append(res['p2'])
            c1.append(res['c1'])
            c2.append(res['c2'])
            i1+=1
        blabels = np_utils.to_categorical(np.array(blabels), 2)

        result = [[b3, b4], blabels]
        if self.with_paths:
            result += [[p1,p2]]

        return result

def load_img_temporal(img, vec_size2, tam, metadata_dict):
  iplt2 = [process_load(img[1][i], vec_size2, None) for i in range(tam)]
  iplt3 = [process_load(img[3][i], vec_size2, None) for i in range(tam)]

  d1 = {"i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":str(img[0]),
        "p2":str(img[2]),
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }

  d1['metadata'] = []
  for i in range(tam):
    diff = abs(np.array(metadata_dict[img[0][i]][:7]) - np.array(metadata_dict[img[2][i]][:7])).tolist()
    for j in range(len(diff)):
      diff[j] = 1 if diff[j] else 0
    d1['metadata'] += metadata_dict[img[0][i]] + metadata_dict[img[2][i]] + diff
  d1['metadata'] = np.array(d1['metadata'])
  return d1

class SiameseSequenceTemporal(Sequence):
    def __init__(self,features, 
                augmentations,
                tam, 
                metadata_dict, 
                metadata_length, 
                batch_size,
                with_paths=False):
        self.tam = tam
        self.features = features
        self.batch_size = batch_size
        self.vec_size2 = (image_size_h_c,image_size_w_c,nchannels)
        self.metadata_dict = metadata_dict
        self.metadata_length = metadata_length
        self.augment = augmentations
        self.with_paths = with_paths

    def __len__(self):
        return int(np.ceil(len(self.features) / 
            float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = self.features[start:end]
        futures = []
        _vec_size2 = (len(batch),self.tam,) + self.vec_size2
        b3 = np.zeros(_vec_size2)
        b4 = np.zeros(_vec_size2)
        blabels = np.zeros((len(batch)))
        p1 = []
        p2 = []
        c1 = []
        c2 = []
        if self.metadata_length>0:
            metadata = np.zeros((len(batch),self.metadata_length))
        i = 0
        for _b in batch:
            r = load_img_temporal(_b, self.vec_size2, self.tam, self.metadata_dict)
            for j in range(self.tam):
                b3[i,j,:,:,:] = self.augment[2][j](image=r['i2'][j])["image"]
                b4[i,j,:,:,:] = self.augment[3][j](image=r['i3'][j])["image"]
            blabels[i] = r['l']
            p1.append(r['p1'])
            p2.append(r['p2'])
            c1.append(r['c1'])
            c2.append(r['c2'])
            if self.metadata_length>0:
                metadata[i,:] = r['metadata']
            i+=1
        blabels2 = np.array(blabels).reshape(-1,1)
        blabels = np_utils.to_categorical(blabels2, 2)
        y = {"class_output":blabels, "reg_output":blabels2}
        result = [[b3, b4, metadata], y]

        if self.with_paths:
          result += [[p1,p2]]

        return result