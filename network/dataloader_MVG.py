import os,re
import numpy as np
import tensorflow as tf
import argparse
import importlib
import glob, random
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

class project_params:
    def __init__(self, params):
        self.params = params
    
def add_arguments(parser):
    parser.add_argument('--data_path', type=str, default='./parent/directory/to/the/main/data/path')
    parser.add_argument('--mice_flist', type=str, default='./config.txt', help='configuration txt file containing the mouse name used for training and validation')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used in training')
    parser.add_argument('--seed', type=int, default=42, help='a defined seed used globally')
    parser.add_argument('--num_classes', type=int, default=3, help='2/3/4')
    parser.add_argument('--test_mice', type=str, default='191204', help='the name of the test mice')
    parser.add_argument('--brain_area', type=str, default='whole', help='number of parcels used in generating MVG')
    parser.add_argument('--loss',type=str,default='categorical', help='categorical (cross-entropy loss)/focal (focal loss)')
    parser.add_argument('--timelen', type=int, default=10, help='epoch duration in seconds')

    return parser

class dataloader:
    def __init__(self, project):
        self.project = project
        # create single separate txt file that contains mouse names used in both training and validation
        self.mice_flist = open(f"./{self.project.params.mice_flist}.txt", "r").read().splitlines()
        self.project.params.num_frames = int(np.floor(self.project.params.timelen * 16.8))
        print(f"Using {self.project.params.num_frames} frames\n")
        self.project.params.brain_idx = get_brain_areas(self.project.params.brain_area)
        if self.project.params.num_classes == 2:
            self.states = ['Awake', 'NREM']
        elif self.project.params.num_classes == 3:
            self.states = ['Awake', 'NREM', 'REM']
        elif self.project.params.num_classes == 4:
            self.states = ['Awake', 'NREM', 'KX', 'Movement']

    def load_data(self):
        
        if self.project.params.mode == 'test_subjectwise' or self.project.params.mode == 'gradcam': 
            x_test, y_test = self.load_data_subjectwise()
        
        elif self.project.params.num_classes == 2:
            x, y, fns = self.load_data_cross_dataset()
            x_train, y_train, fn_train, x_val, y_val, fn_val, x_test, y_test, fn_test = self.split_data(x, y, fns)
            print('num_train:%d, num_val:%d, num_test:%d'%(len(x_train), len(x_val), len(x_test)))

        elif self.project.params.num_classes == 3:
            x, y, fns = self.load_data_2020()
            x_train, y_train, fn_train, x_val, y_val, fn_val, x_test, y_test, fn_test = self.split_data(x, y, fns)
            write_fnames(fn_train, fn_val, fn_test, self.project.params.mice_flist)
            print('num_train:%d, num_val:%d, num_test:%d'%(len(x_train), len(x_val), len(x_test)))
        
        elif self.project.params.num_classes == 4:
            x, y, fns = self.load_data_2016()
            x_train, y_train, fn_train, x_val, y_val, fn_val, x_test, y_test, fn_test = self.split_data(x, y, fns)
            print('num_train:%d, num_val:%d, num_test:%d'%(len(x_train), len(x_val), len(x_test)))
        
        if self.project.params.mode == 'train':
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=128, seed=None, reshuffle_each_iteration=True).batch(self.project.params.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_dataset = val_dataset.shuffle(buffer_size=128, seed=None, reshuffle_each_iteration=False).batch(self.project.params.batch_size, drop_remainder=True)
            
            print('... Training and validation data loaded ...')
            return train_dataset, val_dataset

        elif self.project.params.mode == 'test' or self.project.params.mode == 'gradcam':
             
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            test_dataset = test_dataset.batch(self.project.params.batch_size)
            
            print('... Test data loaded ...')
            return test_dataset

        elif self.project.params.mode == 'test_subjectwise':
            test_dataset = tf.data.Dataset.from_tensor_slices((x, y))   
            test_dataset = test_dataset.batch(self.project.params.batch_size)
 
            print('... Test data loaded ...')
            return test_dataset
   
    def load_data_2020(self):
        # this function loads the sleep dataset
        x, y, fns = [], [], []
        
        for mice in self.mice_flist:
            # check the main data path pointing conatining  MVG folders
            fnames = sorted(glob.glob(os.path.join(self.project.params.data_path, "2020-G5", f"{mice}-MVG", "*.mat")))
            if fnames:
                print(f"Glob {mice} data")

            for f in fnames:
                data = sio.loadmat(f)
                adjacency = np.transpose(data['am'])
                label = data['label'][0]
                x.append(adjacency[:self.project.params.num_frames, :self.project.params.num_frames, self.project.params.brain_idx])
                y.append(label)
                fns.append(os.path.splitext(os.path.basename(f))[0])

        if 'focal' in self.project.params.loss:
            y = np.argmax(label_binarize(y, classes=[0,1,2]), axis=1)
        else: 
            y = label_binarize(y, classes=[0,1,2])
        return x, y, fns
     
    def load_data_2016(self):
        # this function loads the dataset containing other states of consciouness
        x, y, fns = [], [], []
        
        for state in self.states:
            # check the main data path pointing conatining MVG folders
            fnames = sorted(glob.glob(os.path.join(self.project.params.data_path, "2016-G5", "MVG", state, '*.mat')))
            if fnames:
                print(f"Glob 2016 {state} data")

            for f in fnames:
                adjacency = np.transpose(sio.loadmat(f)['am'])
                x.append(adjacency)
                y.append(state)
                fns.append(os.path.splitext(os.path.basename(f))[0])

        if 'focal' in self.project.params.loss and self.project.params.num_classes != 2:
            y = np.argmax(label_binarize(y, classes=self.states), axis=1)
        else: 
            y = label_binarize(y, classes=self.states)

        print(y)
        return x, y, fns
    
    def load_data_cross_dataset(self):

        x1, y1, fns1 = self.load_data_2016()
        x2, y2, fns2 = [], [], []
        
        for mice in self.mice_flist:
            fnames = sorted(glob.glob(os.path.join(self.project.params.data_path, "2020-G5", f"{mice}-MVG", "*.mat")))
            if fnames:
                print(f"Glob {mice} data")

            for f in fnames:
                data = sio.loadmat(f)
                label = int(data['label'][0])
                adjacency = np.transpose(data['am'])
                if label == 2:
                    label = 1
                x2.append(adjacency)
                y2.append(label)
                fns2.append(os.path.splitext(os.path.basename(f))[0])
         
        y2 = label_binarize(y2, classes=[0,1])
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        fns = np.concatenate((fns1, fns2), axis=0)

        return x, y, fns
  

    def load_data_subjectwise(self):
        
        x,y = [], []
        
        # check the main data path pointing conatining  MVG folders
        self.project.params.data_path = os.path.join(self.project.params.data_path, "2020-G5", f"{self.project.params.test_mice}-MVG")
        files = sorted(glob.glob(os.path.join(self.project.params.data_path, "*.mat")), key=lambda x: get_regexp(x, num_key=2))
         
        for file in files:
            data = sio.loadmat(file)
            adjacency = np.transpose(data['am'])
            label = int(data['label'][0])
            x.append(adjacency[:self.project.params.num_frames, :self.project.params.num_frames, self.project.params.brain_idx])
            y.append(label)

        if 'focal' in self.project.params.loss:
            y = np.argmax(label_binarize(y, classes=[0,1,2]), axis=1)
        else:
            y = label_binarize(y, classes=[0,1,2])
        
        return x, y

    def split_data(self, x, y, fns=None):
        seed = self.project.params.seed
        x_train, x_tmp, y_train, y_tmp, fn_train, fn_tmp = train_test_split(x, y, fns, test_size=0.2, random_state=seed, shuffle=True)
        x_val, x_test, y_val, y_test, fn_val, fn_test = train_test_split(x_tmp, y_tmp, fn_tmp, test_size=0.5, random_state=seed, shuffle=True)

        return x_train, y_train, fn_train, x_val, y_val, fn_val, x_test, y_test, fn_test

def write_fnames(fn_train, fn_val, fn_test, mice_flist):
    config_name = os.path.splitext(mice_flist)[0]
    with open(f"./{config_name}_train_fnames_binary.txt", "wt") as train_txt:
        train_txt.write('\n'.join(fn_train))
    with open(f"./{config_name}_val_fnames_binary.txt", "wt") as val_txt:
        val_txt.write('\n'.join(fn_val))
    with open(f"./{config_name}_test_fnames_binary.txt", "wt") as test_txt:
        test_txt.write('\n'.join(fn_test))
   
def get_regexp(fname, num_key=2):
    # change the name pattern for your own filenames
    m = re.match(r"(\d+)-(\w+\d+)-fc(\d+)-GSR_G5_epoch(\d+).mat", os.path.basename(fname))
    name_parts = (m.groups())
    if num_key == 2:
        return int(name_parts[2]), int(name_parts[3])
    elif num_key == 1:
        return int(name_parts[2])

def get_brain_areas(brain_area):
    if brain_area == 'whole':
        area_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    else:
        area_idx = [int(brain_area)]
    return area_idx
