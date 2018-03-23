import threading
import time
import logging
import random
import Queue
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

EPS = 1e-5

def print_loss(train_hist,val_hist,result_dir):
    ################################################
    # Plot loss
    ################################################
    #loss curve
    plt.figure()
    plt.plot(train_hist, linewidth=2,color='r',label='train')
    plt.plot(val_hist, linewidth=2,color='g',label='val')
    plt.legend()
    plt.savefig(result_dir+'/loss.pdf',dpi=600)
    plt.show()
    plot.close()
    return True


def random_crop(image_pair,max_offset,crop_dims):
    """assumes image_pair is a list of 2d arrays with same shape"""
    starts = np.random.randint(max_offset, size=2)*2 - max_offset
    W,H = image_pair[0].shape
    startsx = W/2-crop_dims/2+starts[0]
    startsy = H/2-crop_dims/2+starts[1]

    return_images = []
    for i,im in enumerate(image_pair):

        return_images.append(
            im[startsx:startsx+crop_dims,startsy:startsy+crop_dims])

    return tuple(return_images)
'''
def random_rotate(image_pairList):
    angle = np.random.randint(360)
    return_imagesList =[]
    for count in range(len(image_pairList)):
        return_images = []
	print 'image pair:'
	print count
        for i,im in enumerate(image_pairList[count]):
            return_images.append(rotate(im,angle,axes=(1,0),reshape=False))
        return_imagesList.append(return_images)
    return tuple(return_imagesList)
'''
def random_rotate(image_pair, angle):
    return_images = []
    for i,im in enumerate(image_pair):
        return_images.append(rotate(im,angle,axes=(1,0),reshape=False))

    return tuple(return_images)

class FileReaderThread(threading.Thread):
    """Note this class is a thread, so it runs in a separate thread parallel
    to the main program"""
    def __init__(self, q, file_list, batch_ids, path_start, path_length, num_batch, reader_fn, group=None, target=None, name="producer",
                 args=(), kwargs=None, verbose=None):
        super(FileReaderThread,self).__init__()
        self.target    = target
        self.name      = name
        self.file_list = file_list
        self.reader_fn = reader_fn
	self.batch_ids = batch_ids
        self.path_length=path_length
        self.path_start= path_start
        self.num_batch= num_batch
        self.q         = q

    def run(self):
        while True:
            if not self.q.full():
		'''
		file_ = np.random.choice(self.file_list)
                item_ = self.reader_fn(file_)
                self.q.put(item_)
                time.sleep(random.random())
		'''
                #fill queue with Lists of images along random paths of the same lenght building a batch
                batchList = np.random.choice(self.batch_ids)
                self.num_batch=len(batchList)
        	itemList=[]
        	for iterator in range(len(batchList)):
                    for it in range(self.path_length[batchList[iterator]]):
                        fileInput=np.array(self.reader_fn(self.file_list[self.path_start[batchList[iterator]]+it]))
			itemList.append(fileInput)
                    self.q.put(itemList)
                time.sleep(random.random())
        return

class BatchGetter(object):
    def __init__(self, preprocessor_fn, batch_processor_fn, num_batch, batch_ids, path_start, path_length, queue_size, file_list, reader_fn, num_threads=1):
        self.q                  = Queue.Queue(queue_size)
        self.preprocessor_fn    = preprocessor_fn
        self.batch_processor_fn = batch_processor_fn
        self.num_batch          = num_batch
        self.file_list          = file_list
	self.batch_ids	    	= batch_ids
        self.path_length        = path_length
        self.path_start         = path_start
        self.reader_fn          = reader_fn
        self.num_threads        = num_threads

        self.readers = []
        for i in range(self.num_threads):
            t = FileReaderThread(self.q,self.file_list,self.batch_ids, self.path_start, self.path_length, self.num_batch, self.reader_fn,name='producer'+str(i))
            t.setDaemon(True)
            t.start()
            self.readers.append(t)

    def get_batch(self):
        items = []
        while len(items) < self.num_batch:
            item_ = self.q.get()
	    #print item_
            item_ = self.preprocessor_fn(item_)
            items.append(item_)
            #returns a list of lists of image tuples along a vessec path of the same length
        return self.batch_processor_fn(items)
	#return items
