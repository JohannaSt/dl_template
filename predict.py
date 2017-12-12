from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
from modules import train_utils
import os
import Queue
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import argparse
import scipy
import importlib

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('global_config_file')
parser.add_argument('case_config_file')
parser.add_argument('--val')
parser.add_argument('--test')

args = parser.parse_args()

global_config_file = os.path.abspath(args.global_config_file)
case_config_file = os.path.abspath(args.case_config_file)
val  = args.val
test = args.test


global_config = io.load_yaml(global_config_file)
case_config   = io.load_yaml(case_config_file)

try:
    sv.mkdir(case_config['RESULTS_DIR'])
except:
    raise RuntimeError("Unable to create results directory {}".format(case_config['RESULTS_DIR']))

#########################
# Get train/val files
#########################
if val:
    files = open(case_config['VAL_FILE_LIST'],'r').readlines()
    files = [f.replace('\n','') for f in files]
    OUTPUT_DIR = case_config['RESULTS_DIR']+'/val'
    sv.mkdir(OUTPUT_DIR)
    print "writing validation files to {}".format(OUTPUT_DIR)
elif test:
    files = open(case_config['TEST_FILE_LIST'],'r').readlines()
    files = [f.replace('\n','') for f in files]
    OUTPUT_DIR = case_config['RESULTS_DIR']+'/test'
    sv.mkdir(OUTPUT_DIR)
    print "writing test files to {}".format(OUTPUT_DIR)
else:
    raise RuntimeError("must specify --val or --test as argument")

##########################
# Import experiment
##########################
experiment = importlib.import_module(case_config['EXPERIMENT_FILE'])

model  = experiment.Model(global_config, case_config)

model.load()

reader = experiment.read_file

preprocessor    = experiment.normalize

batch_processor = experiment.tuple_to_batch

evaluator       = experiment.evaluate
##########################
# Calculate Error
##########################
CD = global_config['CROP_DIMS']
ID = global_config['IMAGE_DIMS']
TH = global_config['THRESHOLD']

for f in tqdm(files):
    xb,yb = reader(f)

    xb = xb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    yb = yb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]

    T  = preprocessor((xb,yb), case_config)
    T  = batch_processor(T)

    yp       = model.predict(T[0])[0,:,:,0]
    err_dict, yp_thresh = evaluator(T, model, global_config)
    err_dict['GROUND_TRUTH'] = f

    image_name = f.split('/')[-3]
    path_name  = f.split('/')[-2]
    point_number  = f.split('/')[-1]

    ofn = OUTPUT_DIR+"/{}.{}.{}".format(image_name,path_name,point_number)
    ofn_csv = ofn+'.csv'
    ofn_np  = ofn+'.ypred.npy'

    err_dict['PREDICTION'] = ofn_np

    scipy.misc.imsave(ofn+'.x.png',xb)
    scipy.misc.imsave(ofn+'.ypred.png',yp)
    scipy.misc.imsave(ofn+'.y.png',yb)
    scipy.misc.imsave(ofn+'.ypred_thresh.png',yp_thresh)

    io.write_csv(ofn_csv,err_dict)
    np.save(ofn_np,yp)
