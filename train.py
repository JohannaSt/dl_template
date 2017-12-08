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
        x = (1.0*x-np.amin(x))/(np.amax(x)-np.amin(x)+1e-5)
        image_pair[0] = x
        y = image_pair[1]
        y = (1.0*y-np.amin(y))/(np.amax(y)-np.amin(y)+1e-5)
        image_pair[1] = y

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
    y = np.round(y)
    return x,y

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
leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)

#I2INetFC
yclass,yhat = tf_util.I2INetFC(x, nfilters=NUM_FILTERS, activation=leaky_relu)

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
model_dir   = case_config['MODEL_DIR']
sv.mkdir(batch_dir)
sv.mkdir(model_dir)

if case_config.has_key('PRETRAINED_MODEL_PATH'):
    saver.restore(sess,case_config['PRETRAINED_MODEL_PATH'])

train_hist = []
val_hist   = []
for i in range(TRAIN_STEPS+1):
    global_step = global_step+1
    xb,yb = consumer.get_batch()

    if np.sum(np.isnan(xb)) > 0: continue
    if np.sum(np.isnan(yb)) > 0: continue

    l,_ = sess.run([loss,train],{x:xb,y:yb})

    if i%PRINT_STEP == 0:
        fval = np.random.choice(val_files)
        xv,yv = preprocessor(reader(fval))
        xv = xv[np.newaxis,:,:,np.newaxis]
        yv = yv[np.newaxis,:,:,np.newaxis]

        lval = sess.run(loss,{x:xv,y:yv})
        ypred,yb,xb = sess.run([yclass,y,x],{x:xb,y:yb})

        train_hist.append(l)
        val_hist.append(lval)

        print "{}: train_loss={}, val_loss={}".format(i,l,lval)

        saver.save(sess,model_dir+'/{}'.format(case_config['MODEL_NAME']))

        for j in range(global_config["BATCH_SIZE"]):

            plt.figure()
            plt.imshow(xb[j,:,:,0],cmap='gray')
            plt.colorbar()
            plt.savefig('{}/{}.{}.x.png'.format(batch_dir,i,j))
            plt.close()

            plt.figure()
            plt.imshow(yb[j,:,:,0],cmap='gray')
            plt.colorbar()
            plt.savefig('{}/{}.{}.y.png'.format(batch_dir,i,j))
            plt.close()

            plt.figure()
            plt.imshow(ypred[j,:,:,0],cmap='gray')
            plt.colorbar()
            plt.savefig('{}/{}.{}.ypred.png'.format(batch_dir,i,j))
            plt.close()

        plt.figure()
        plt.plot(train_hist,color='r',label='train')
        plt.plot(val_hist,color='g',label='val')
        plt.legend()
        plt.savefig('{}/{}.loss.png'.format(batch_dir,i))
        plt.close()
