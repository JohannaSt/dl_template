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

try:
    sv.mkdir(case_config['RESULTS_DIR'])
except:
    raise RuntimeError("Unable to create results directory {}".format(case_config['RESULTS_DIR']))

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
    return [x,y]

def preprocessor(image_pair):
    """ images is a [x,y] pair """
    if case_config['LOCAL_MAX_NORM']:
        x = image_pair[0]
        x = (1.0*x-np.amin(x))/(np.amax(x)-np.amin(x))
        image_pair[0] = x
        y = image_pair[1]
        y = (1.0*y-np.amin(y))/(np.amax(y)-np.amin(y))
        image_pair[1] = y.astype(int)

    if case_config['ROTATE']:
        image_pair = train_utils.random_rotate(image_pair)

    if case_config['RANDOM_CROP']:
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

###############################
# Set up variable learning rate
###############################
LEARNING_RATE = global_config["LEARNING_RATE"]
global_step = tf.Variable(0, trainable=False)
boundaries = [5000, 10000]
values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100]
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
leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)

Nbatch = tf.shape(x)[0]

#I2INet
yclass,yhat,o3,o4 = tf_util.I2INet(x,nfilters=NUM_FILTERS,
    activation=leaky_relu,init=INIT)

#I2INetFC
y_vec = tf.reshape(yhat, (Nbatch,CROP_DIMS**2))

sp = tf_util.fullyConnected(y_vec,CROP_DIMS,leaky_relu, std=INIT, scope='sp1')
sp = tf_util.fullyConnected(y_vec,CROP_DIMS**2,leaky_relu, std=INIT, scope='sp2')
sp = tf.reshape(sp, (Nbatch,CROP_DIMS,CROP_DIMS,1))

y_sp = tf_util.conv2D(sp, nfilters=NUM_FILTERS,
    activation=leaky_relu,init=INIT, scope='sp3')
y_sp_1 = tf_util.conv2D(y_sp, nfilters=NUM_FILTERS,
    activation=leaky_relu, init=INIT,scope='sp4')
y_sp_2 = tf_util.conv2D(y_sp_1, nfilters=NUM_FILTERS,
    activation=leaky_relu, init=INIT,scope='sp5')

yhat = tf_util.conv2D(y_sp_2, nfilters=1, activation=tf.identity, init=INIT,scope='sp6')

yclass = tf.sigmoid(yhat)

#Loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat,labels=y))

loss = loss + tf_util.l2_reg(LAMBDA)

opt = tf.train.AdamOptimizer(learning_rate)
train = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print yclass

saver = tf.train.Saver()

##############################
# Train
##############################
TRAIN_STEPS = global_config['TRAIN_STEPS']
PRINT_STEP  = global_config['PRINT_STEP']
batch_dir   = case_config['RESULTS_DIR']+'/batch'
sv.mkdir(batch_dir)

if case_config.has_key('PRETRAINED_MODEL_PATH'):
    saver.restore(sess,case_config['PRETRAINED_MODEL_PATH'])

train_hist = []
val_hist   = []
for i in range(TRAIN_STEPS+1):
    global_step = global_step+1
    xb,yb = consumer.get_batch()

    l,_ = sess.run([loss,train],{x:xb,y:yb})

    if i%PRINT_STEP == 0:
        fval = np.random.choice(val_files)
        xv,yv = preprocessor(reader(fval))
        xv = xv[np.newaxis,:,:,np.newaxis]
        yv = yv[np.newaxis,:,:,np.newaxis]

        lval,ypred = sess.run([loss,yclass],{x:xv,y:yv})

        train_hist.append(l)
        val_hist.append(lval)

        print "{}: train_loss={}, val_loss={}".format(i,l,lval)
