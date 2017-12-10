import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
This file builds the I2INet network and sets up the required

*file reader
*file post processor
*trainer
*saver
*predictor
*logger

The idea is that by keeping these definitions in a separate file
we can make the choice of network configurable by simply referencing
a particular file
"""
def build_model(global_config,case_config):
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
    
def read_file(filename):
    x = np.load(filename+'.X.npy')
    y = np.load(filename+'.Yc.npy')
    return (x,y)

def normalize(Tuple, case_config):
    x = Tuple[0]
    x = (1.0*x-np.amin(x))/(np.amax(x)-np.amin(x)+1e-5)
    
    y = Tuple[1]
    y = (1.0*y-np.amin(y))/(np.amax(y)-np.amin(y)+1e-5)
    return (x,y)
    
def augment(Tuple, case_config):
    if case_config['ROTATE']:
        Tuple = train_utils.random_rotate(Tuple)

    if case_config['RANDOM_CROP']:
        Tuple = train_utils.random_crop(Tuple,case_config['PATH_PERTURB'],
            global_config['CROP_DIMS'])
    
    return Tuple

def tuple_to_batch(tuple_list):
    if type(tuple_list) == list and len(tuple_list) == 1:
        tuple_list = tuple_list[0]
    if type(tuple_list) == tuple:
        x = tuple_list[0]
        x = x[np.newaxis,:,:,np.newaxis]
        y = tuple_list[1]
        y = y[np.newaxis,:,:,np.newaxis]
        return x,y
    else:
        x = np.stack([pair[0] for pair in tuple_list])
        x = x[:,:,:,np.newaxis]

        y = np.stack([pair[1] for pair in tuple_list])
        y = y[:,:,:,np.newaxis]
        y = np.round(y)
        return x,y

def train_step(Tuple):
    global_step = global_step+1
    xb,yb = Tuple

    if np.sum(np.isnan(xb)) > 0: return
    if np.sum(np.isnan(yb)) > 0: return

    sess.run(train,{x:xb,y:yb})
    
def save_model():
    saver.save(sess,model_dir+'/{}'.format(case_config['MODEL_NAME']))
    
def predict(Tuple):
    
    
def log(train_tuple, val_tuple):
    xb,yb = train_tuple
    xv,yv = val_tuple
    
    lval = sess.run(loss,{x:xv,y:yv})
    ypred,yb,xb = sess.run([yclass,y,x],{x:xb,y:yb})

    train_hist.append(l)
    val_hist.append(lval)

    print "{}: train_loss={}, val_loss={}".format(i,l,lval)

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
