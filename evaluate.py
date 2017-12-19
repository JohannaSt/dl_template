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
import pandas as pd
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
    print "evaluating validation predictions {}".format(OUTPUT_DIR)
elif test:
    files = open(case_config['TEST_FILE_LIST'],'r').readlines()
    files = [f.replace('\n','') for f in files]
    OUTPUT_DIR = case_config['RESULTS_DIR']+'/test'
    sv.mkdir(OUTPUT_DIR)
    print "evaluating test predictions {}".format(OUTPUT_DIR)
else:
    raise RuntimeError("must specify --val True or --test True as argument")

##########################
# Import experiment
##########################
experiment = importlib.import_module(case_config['EXPERIMENT_FILE'])

model  = experiment.Model(global_config, case_config)

model.load()

reader = experiment.read_file

preprocessor    = experiment.normalize

batch_processor = experiment.tuple_to_batch

evaluator       = experiment.calculate_error

threshold_array = np.arange(0,1,1.0/global_config['PR_INTERVALS'])

df = pd.DataFrame()

CD = global_config['CROP_DIMS']
ID = global_config['IMAGE_DIMS']

for f in tqdm(files):
    Tuple = reader(f)
    xb = Tuple[0]
    yb = Tuple[1]
    yc = Tuple[2]

    xb = xb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    yb = yb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    yc = yc[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]

    T  = preprocessor((xb,yb), case_config)
    yb = T[1]

    point_number = f.split('/')[-1]
    path_name    = f.split('/')[-2]
    image_name   = f.split('/')[-3]

    prediction_prefix = OUTPUT_DIR+'/{}.{}.{}'.format(image_name,path_name,point_number)
    p_csv = prediction_prefix+'.csv'
    err_dict = io.read_csv(p_csv)

    ypred = np.load(err_dict['PREDICTION'])

    for thresh in threshold_array:
        yp = ypred.copy()
        yp[yp<thresh]  = 0
        yp[yp>=thresh] = 1
        if np.sum(yp) < 1:
            yp[ID/2,ID/2] = 1
        err_dict = evaluator(yp,yb)

        err_dict['IMAGE']      = image_name
        err_dict['PATH_NAME']  = path_name
        err_dict['PATH_POINT'] = point_number
        err_dict['THRESHOLD']  = thresh

        df = df.append(err_dict,ignore_index=True)

df.to_csv(OUTPUT_DIR+'/pr.csv')
