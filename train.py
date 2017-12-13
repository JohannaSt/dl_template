from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
from modules import train_utils

import os

import Queue

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import importlib

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
experiment = importlib.import_module(case_config['EXPERIMENT_FILE'],'pkg.subpkg')

model  = experiment.Model(global_config, case_config)

reader = experiment.read_file

def make_preprocessor(global_config, case_config):

    def preprocess(Tuple):
        Tuple = experiment.normalize(Tuple, case_config)
        Tuple = experiment.augment(Tuple, global_config, case_config)
        return Tuple

    return preprocess

preprocessor    = make_preprocessor(global_config, case_config)

batch_processor = experiment.tuple_to_batch

logger          = experiment.log

##########################
# Setup queues and threads
##########################
consumer = train_utils.BatchGetter(preprocessor,batch_processor,global_config['BATCH_SIZE'],
queue_size=global_config['QUEUE_SIZE'], file_list=train_files,
reader_fn=reader, num_threads=global_config['NUM_THREADS'])

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
   model.load(model_path=case_config['PRETRAINED_MODEL_PATH'])
   print "loaded model"

train_hist = []
val_hist   = []
print "Starting train loop"
for i in range(TRAIN_STEPS+1):

    train_tuple = consumer.get_batch()
    #train step
    model.train_step(train_tuple)

    if i%PRINT_STEP == 0:
        fval = np.random.choice(val_files)
        val_tuple = preprocessor(reader(fval))
        val_tuple = batch_processor(val_tuple)
        #log stuff

        l_train, l_val, _ = logger(train_tuple, val_tuple, model, case_config, i)

        print "{}: train loss = {}, val loss = {}".format(i,l_train, l_val)

        model.save()
