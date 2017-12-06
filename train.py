from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
from modules import train_utils
import os
import Queue
import numpy as np

import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('global_config_file')
parser.add_argument('case_config_file')

args = parser.parse_args()

global_config_file = os.path.abspath(args.global_config_file)
case_config_file = os.path.abspath(args.case_config_file)


global_config = io.load_yaml(global_config_file)
case_config   = io.load_yaml(case_config_file)

#########################
# Get train/val files
#########################
files_ = open(case_config['DATA_DIR']+'/files.txt','r').readlines()
files_ = [f.replace('\n','') for f in files_]

train_files = [f for f in files_ if any(k in f for k in case_config['TRAIN_IMAGES'])]
val_files   = [f for f in files_ if any(k in f for k in case_config['VAL_IMAGES'])]

##########################
# build augmenter
##########################
def reader(filename):
    x = np.load(filename+'.X.npy')
    y = np.load(filename+'.Yc.npy')
    print x.shape,y.shape
    return [x,y]

def preprocessor(image_pair):
    """ images is a [x,y] pair """
    if case_config['LOCAL_MAX_NORM']:
        print "normalizing"
        x = image_pair[0]
        x = (1.0*x-np.amin(x))/(np.amax(x)-np.amin(x))
        image_pair[0] = x
        y = image_pair[1]
        y = (1.0*y-np.amin(y))/(np.amax(y)-np.amin(y))
        image_pair[1] = y.astype(int)

    if case_config['ROTATE']:
        print "rotating"
        image_pair = train_utils.random_rotate(image_pair)

    if case_config['RANDOM_CROP']:
        print "cropping"
        image_pair = train_utils.random_crop(image_pair,case_config['PATH_PERTURB'],
            global_config['CROP_DIMS'])

    return image_pair

def batch_processor(im_list):
    x = np.stack([pair[0] for pair in im_list])
    x = x[:,:,:,np.newaxis]

    y = np.stack([pair[1] for pair in im_list])
    y = y[:,:,:,np.newaxis]

    return x,y
##########################
# Setup queues and threads
##########################
Q        = Queue.Queue(global_config['QUEUE_SIZE'])
producer = train_utils.FileReaderThread(Q,train_files, reader)
producer.setDaemon(True)
consumer = train_utils.BatchGetter(Q,preprocessor,batch_processor,global_config['BATCH_SIZE'])
producer.start()
