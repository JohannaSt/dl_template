from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
from modules import train_utils
import os
import Queue
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from medpy.metric.binary import hd, assd
import csv
import argparse
import scipy
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
leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)

#I2INetFC
yclass,yhat = tf_util.I2INetFC(x, nfilters=NUM_FILTERS, activation=leaky_relu)

sess = tf.Session()

saver = tf.train.Saver()

saver.restore(sess,case_config['MODEL_DIR']+'/'+case_config['MODEL_NAME'])

##########################
# Calculate Error
##########################
CD = global_config['CROP_DIMS']
ID = global_config['IMAGE_DIMS']
TH = global_config['THRESHOLD']

def calculate_error(ypred,y):
    """assumes ypred and y are thresholded"""
    TP = np.sum(ypred*y)
    FP = np.sum(ypred*(1-y))
    TN = np.sum((1-ypred)*(1-y))
    FN = np.sum((1-ypred)*y)
    HD = hd(y,ypred)
    ASSD = assd(y,ypred)
    DICE = (1.0*TP)/(TP+FN)
    return {"TP":TP, "FP":FP, "TN":TN, "FN":FN, "HD":HD, "ASSD":ASSD, "DICE":DICE}

def write_csv(filename,dict):
    with open(filename,'w') as f:
        w = csv.DictWriter(f,dict.keys())
        w.writeheader()
        w.writerow(dict)

for f in tqdm(files):
    xb = np.load(f+'.X.npy')
    xb = xb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    xb = (1.0*xb-np.amin(xb))/(np.amax(xb)-np.amin(xb)+1e-5)
    xb = xb[np.newaxis,:,:,np.newaxis]

    yb = np.load(f+'.Yc.npy')
    yb = yb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    yb = (1.0*yb-np.amin(yb))/(np.amax(yb)-np.amin(yb)+1e-5)
    if np.sum(yb) < 1:
        yb[ID/2,ID/2] = 1
    yp = sess.run(yclass,{x:xb})
    yp = yp[0,:,:,0]

    yp_thresh = yp.copy()
    yp_thresh[yp_thresh > TH] = 1
    yp_thresh[yp_thresh <= TH] = 0
    if np.sum(yp_thresh) < 1:
        yp_thresh[ID/2,ID/2] = 1
    err_dict = calculate_error(yp_thresh.astype(int),np.round(yb).astype(int))

    image_name = f.split('/')[-3]
    path_name  = f.split('/')[-2]
    point_number  = f.split('/')[-1]

    ofn = OUTPUT_DIR+"/{}.{}.{}".format(image_name,path_name,point_number)
    ofn_csv = ofn+'.csv'
    ofn_np  = ofn+'.ypred.npy'

    scipy.misc.imsave(ofn+'.x.png',xb[0,:,:,0])
    scipy.misc.imsave(ofn+'.ypred.png',yp)
    scipy.misc.imsave(ofn+'.y.png',yb)
    scipy.misc.imsave(ofn+'.ypred_thresh.png',yp_thresh)

    write_csv(ofn_csv,err_dict)
    np.save(ofn_np,yp)
