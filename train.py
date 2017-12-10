from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
from modules import train_utils
import os
import Queue
import numpy as np
import matplotlib.pyplot as plt
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

try:
    sv.mkdir(case_config['RESULTS_DIR'])
except:
    raise RuntimeError("Unable to create results directory {}".format(case_config['RESULTS_DIR']))

#########################
# Get train/val files
#########################
train_files = open(case_config['TRAIN_FILE_LIST'],'r').readlines()
train_files = [f.replace('\n','') for f in train_files]

val_files = open(case_config['VAL_FILE_LIST'],'r').readlines()
val_files = [f.replace('\n','') for f in val_files]

##########################
# import model related functions
##########################


##########################
# Setup queues and threads
##########################
consumer = train_utils.BatchGetter(preprocessor,batch_processor,global_config['BATCH_SIZE'],
queue_size=global_config['QUEUE_SIZE'], file_list=train_files,
reader_fn=reader, num_threads=global_config['NUM_THREADS'])

###############################
# Set up variable learning rate
###############################
LEARNING_RATE = global_config["LEARNING_RATE"]
global_step = tf.Variable(0, trainable=False)
boundaries = [5000, 10000, 15000]
values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100, LEARNING_RATE/1000]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

##########################
# Get Network Parameters
##########################
CROP_DIMS   = global_config['CROP_DIMS']
C           = 1
NUM_FILTERS = global_config['NUM_FILTERS']
LEAK        = global_config['LEAK']
BATCH_SIZE  = global_config['BATCH_SIZE']
INIT        = global_config['INIT']
LAMBDA      = global_config['L2_REG']

##########################
# Build Tensorflow Graph
##########################
#Import lib stuff

##############################
# Train
##############################
TRAIN_STEPS = global_config['TRAIN_STEPS']
PRINT_STEP  = global_config['PRINT_STEP']
batch_dir   = case_config['RESULTS_DIR']+'/batch'
model_dir   = case_config['MODEL_DIR']
sv.mkdir(batch_dir)
sv.mkdir(model_dir)

if case_config.has_key('PRETRAINED_MODEL_PATH'):
    saver.restore(sess,case_config['PRETRAINED_MODEL_PATH'])

train_hist = []
val_hist   = []
for i in range(TRAIN_STEPS+1):

    xb,yb = consumer.get_batch()
    #train step

    if i%PRINT_STEP == 0:
        fval = np.random.choice(val_files)
        xv,yv = preprocessor(reader(fval))
        #log stuff
        