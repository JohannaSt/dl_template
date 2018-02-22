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

#########################
# reshape train/val files
#########################
train_paths = [f.split("/") for f in train_files]
current_path=train_paths[0][-2]
localCounter=0
maxCount=0
globalCounter=0
path_length=[]
while globalCounter<(len(train_paths)-1):
	print globalCounter
	while train_paths[globalCounter][-2]==current_path and globalCounter<(len(train_paths)-1):
		localCounter+=1
		globalCounter+=1
	path_length.append(localCounter+1)
	if localCounter>maxCount:
		maxCount=localCounter+1
	localCounter=0
	current_path=train_paths[globalCounter][-2]
["/".join(f) for f in train_paths]
batch_ids=a = [ [] for i in range(maxCount) ]
for i in range(len(path_length)):
	batch_ids[(path_length[i]-1)].append(i)
print batch_ids

	

val_paths = [f.split("/") for f in val_files]
current_path_val=val_paths[0][-2]
localCounter_val=0
maxCount_val=0
globalCounter_val=0
path_length_val=[]
while globalCounter_val<(len(val_paths)-1):
	while val_paths[globalCounter_val][-2]==current_path_val and globalCounter_val<(len(val_paths)-1):
		localCounter_val+=1
		globalCounter_val+=1
	path_length_val.append(localCounter_val+1)
	if localCounter_val>maxCount_val:
		maxCount_val=localCounter_val+1
	localCounter_val=0
	current_path_val=val_paths[globalCounter_val][-2]
["/".join(f) for f in val_paths]
batch_ids_val=a = [ [] for i in range(maxCount_val) ]
for i in range(len(path_length_val)):
	batch_ids_val[(path_length_val[i]-1)].append(i)
##########################
# import model related functions
##########################
experiment = importlib.import_module(case_config['EXPERIMENT_FILE'],'pkg.subpkg')

model  = experiment.Model(global_config, case_config)

reader = experiment.read_file

def make_preprocessor(global_config, case_config):
    print "make preprocessor"
    def preprocess(TupleList):
	for i in range(len(TupleList)):
		TupleList[i] = experiment.normalize(TupleList[i], case_config)
		TupleList[i] = experiment.augment(TupleList[i], global_config, case_config)
        return TupleList
    return preprocess

preprocessor    = make_preprocessor(global_config, case_config)

batch_processor = experiment.tuple_to_batch

logger          = experiment.log

##########################
# Setup queues and threads
##########################
consumer = train_utils.BatchGetter(preprocessor,batch_processor,global_config['BATCH_SIZE'],
queue_size=global_config['QUEUE_SIZE'], file_list=train_files, batchIDs=batch_ids,
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

#if case_config.has_key('PRETRAINED_MODEL_PATH'):
 #  model.load(model_path=case_config['PRETRAINED_MODEL_PATH'])
  # print "loaded model"

train_hist = []
val_hist   = []
print "Starting train loop"
for i in range(TRAIN_STEPS+1):
    train_tuple = consumer.get_batch()
    #train step
    model.train_step(train_tuple)

    if i%PRINT_STEP == 0:
        
        batchIdsVal = np.random.choice(batch_ids_val)
	batchIdsValList=[]
		for iterator in range(len(batchIdsVal)):
			batchIdsValList.append(reader(val_files[batchIdsVal[iterator]]))
        val_tupleList = preprocessor(batchIdsValList)
        val_tupleList = batch_processor(val_tupleList)
        #log stuff
	#CHANGE!
        l_train, l_val, _ = logger(train_tuple, val_tuple, model, case_config, i)

        print "{}: train loss = {}, val loss = {}".format(i,l_train, l_val)

        train_hist.append(l_train)
        val_hist.append(l_val)

        model.save()
train_utils.print_loss(train_hist,val_hist,case_config['RESULTS_DIR'])
