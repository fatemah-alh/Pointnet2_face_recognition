'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider


type_data='espression'
#ROOT_DIR = os.path.join(ROOT_DIR, 'data')
if type_data=='id':

    rootData="/data/falhamdoosh/face_with_normal"
elif type_data=='espression':
    rootData="/data/falhamdoosh/data_espression"
elif type_data=='unicod':
    rootData="/data/falhamdoosh/data_unicod"
#DATA_DIR=os.path.join(ROOT_DIR, "bosphorusReg3DMM")


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def getDataFiles(list_filename): #get a list of h5filenames (train data divided in 5 h5 files)
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename): # load  h5py file which contain a part of train data modules, 
    f = h5py.File(os.path.join(rootData,h5_filename))
#    print("loadHf file",h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)


class FaceRecognitionDataset(object):
    def __init__(self, list_filename, batch_size = 32, npoints=6704,NUM_CHANNEL=3, shuffle=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints #Number of point in each module (6704,3)
        self.shuffle = shuffle #boolian var 
        self.h5_files = getDataFiles(self.list_filename) #  list of 4 string es.["filenams1.h5",...,"filenams4.h5"] this contain just names of files
        self.NUM_CHANNEL=NUM_CHANNEL
        
        self.reset() # set all variable to 0 and none at the begining of each epoch duting training 

    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))# list with id equal to number of total files h5
        if self.shuffle: np.random.shuffle(self.file_idxs)#change the order of files at each epoch
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0
   
    def _augment_batch_data(self, batch_data): #perform a set of data augmentaion 
        rotated_data = provider.rotate_point_cloud(batch_data[:,:,0:3])
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        batch_data[:,:,0:3] = jittered_data
   #     rotated_data[:,:,3:]
        return provider.shuffle_points(batch_data)


    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]] #return file name h5

    def _load_data_file(self, filename): 
        #load data from one o i fileh5py and associate them to current data and lable
        self.current_data,self.current_label = load_h5(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0 # set batch_idx a 0 at the begining of load of new file
        if self.shuffle:
            self.current_data, self.current_label, _ = shuffle_data(self.current_data,self.current_label)
    
    def _has_next_batch_in_file(self):
                                                       
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return self.NUM_CHANNEL

    def has_next_batch(self):
        # TODO: add backend thread to load data
        #if we are at the begining of th training or at the end of file h5py                                            
        if (self.current_data is None) or (not self._has_next_batch_in_file()): 
            if self.current_file_idx >= len(self.h5_files): #if file is all readed return fals so to stop training
                return False
            #return new current lable and label loading them from new h5py based on self.current_file_idx
            self._load_data_file(self._get_data_filename()) 
            
            self.current_file_idx += 1
        return self._has_next_batch_in_file() # return true or false 

    def next_batch(self, augment=False): #chek augment var 
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        batch_label = np.zeros((bsize), dtype=np.int32)
         #selection a patch from whol data [n module es.32, number of points in module es.6704,all chanell es.3 ]
                                                       
 #       print("npoint",self.npoints )
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :self.NUM_CHANNEL].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if augment: data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch #[32,2048,3]

if __name__=='__main__':
    
    d = FaceRecognitionDataset(os.path.join(rootData ,"trainfiles.txt"))
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
