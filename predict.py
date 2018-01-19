from modules import io
from modules import layers as tf_util
from modules import vascular_data as sv
from modules import train_utils
import os
import Queue
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import argparse
import scipy
import importlib
import pandas as pd
import pickle
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

if case_config['RADIUSRANGEFIXED']:
    evalPerLength = experiment.metricPerLengthFixedRange
else:
    evalPerLength = experiment.metricPerLength
#evalPerLength = experiment.metricPerLength
#evalPerLength = experiment.metricPerLengthFixedRange
##########################
# Calculate Error
##########################
CD = global_config['CROP_DIMS']
ID = global_config['IMAGE_DIMS']
TH = global_config['THRESHOLD']

df = pd.DataFrame()

maxRadius=0
for f in tqdm(files):
    Tuple = reader(f)
    xb = Tuple[0]
    yb = Tuple[1]
    yc = Tuple[2]

    xb = xb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    yb = yb[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]
    yc = yc[ID/2-CD/2:ID/2+CD/2,ID/2-CD/2:ID/2+CD/2]

    T  = preprocessor((xb,yb), case_config)
    T  = batch_processor(T)

    yp       = model.predict(T[0])[0,:,:,0]
    err_dict, yp_thresh, maxRadius, contour = evaluator((T[0],T[1],yc), model, global_config, case_config,  maxRadius)
    err_dict['GROUND_TRUTH'] = f+".Y.npy"

    image_name = f.split('/')[-3]
    path_name  = f.split('/')[-2]
    point_number  = f.split('/')[-1]

    ofn = OUTPUT_DIR+"/{}.{}.{}".format(image_name,path_name,point_number)
    ofn_csv = ofn+'.csv'
    ofn_np  = ofn+'.ypred.npy'

    err_dict['PREDICTION'] = ofn_np

    analysis_name = case_config_file.split('/')[-1]

    err_dict['ANALYSIS_NAME'] = analysis_name
    err_dict['IMAGE'] = image_name
    err_dict['PATH'] = path_name
    err_dict['PATH_POINT'] = point_number

    '''
    plt.figure()
    plt.imshow(yp[:,:],cmap='gray')
    plt.plot(contour[:,0],contour[:,1],color='w',label='contour')
    plt.savefig(ofn+'.ypred_contours.png')
    '''

    scipy.misc.imsave(ofn+'diam'+str(err_dict['RADIUS'])+'.cont.png',contour)
    scipy.misc.imsave(ofn+'.x.png',xb)
    scipy.misc.imsave(ofn+'.ypred.png',yp)
    scipy.misc.imsave(ofn+'.y.png',yb)
    scipy.misc.imsave(ofn+'.ypred_thresh.png',yp_thresh)

    df = df.append(err_dict,ignore_index=True)
    io.write_csv(ofn_csv,err_dict)
    np.save(ofn_np,yp)

df.to_csv(OUTPUT_DIR+'/dataframe.csv')
tablePerLength = pd.DataFrame()
assdaverage=0
hdaverage=0
dcaverage=0
if case_config['RADIUSRANGEFIXED']==True:
    tablePerLength, assdaverage, hdaverage, dcaverage=evalPerLength(df,case_config['RADIUSRANGES'])
else:
    tablePerLength, assdaverage, hdaverage, dcaverage=evalPerLength(df,maxRadius)
#tablePerLength, assdaverage, hdaverage, dcaverage=evalPerLength(df) #,maxRadius)
tablePerLength.to_csv(OUTPUT_DIR+'/metricPerDiameter.csv')

dicePercentage, dicePercentageSmall, dicePercentageLarge, dicePercentagetable =experiment.vesselpercetagePerDiceFixedRange(df,case_config['DICERANGES'],case_config['DICESPLIT'])
dicePercentagetable.to_csv(OUTPUT_DIR+'/DiceThreshold.csv')
print dicePercentage
print dicePercentageSmall
print dicePercentageLarge
#add up the percentages
dicePercentageAdd=dicePercentage+[0.0]
dicePercentageSmallAdd=dicePercentageSmall+[0.0]
dicePercentageLargeAdd=dicePercentageLarge+[0.0]

#dicePercentageAdd[0]=1.0
for j in range(len(dicePercentage)):
    i=len(dicePercentage)-1
    while i>j:
        dicePercentageAdd[j]+=dicePercentage[i]
        dicePercentageSmallAdd[j]+=dicePercentageSmall[i]
        dicePercentageLargeAdd[j]+=dicePercentageLarge[i]
        i-=1
    dicePercentageAdd[j]*=100.0
    dicePercentageSmallAdd[j]*=100.0
    dicePercentageLargeAdd[j]*=100.0
print dicePercentageAdd
print dicePercentageSmallAdd
print dicePercentageLargeAdd
datasave = {'percentageAdd': dicePercentageAdd, 'percentageSmallAdd': dicePercentageSmallAdd,'percentageLargeAdd':dicePercentageLargeAdd}
tablePercentageAdded=pd.DataFrame(data=datasave)
tablePercentageAdded.to_csv(OUTPUT_DIR+'/DiceThresholdPercentage.csv')
#plot percentage of vessels having a certain minimal Dice value
fig, ax = plt.subplots(figsize=(10,10))
ax.set_title('all vessels')
ax.plot(case_config['DICERANGES'],dicePercentageAdd,'o',color='r')
#plt.tick_params(labelsize=11)
ax.set_ylabel('percent of vessels above the Dice threshold',fontsize = 15)
ax.set_xlabel('Dice threshold',fontsize = 15)
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(case_config['DICERANGES'],dicePercentageSmallAdd,'o',color='r')
#plt.tick_params(labelsize=11)
ax.set_title('vessels smaller than: '+str(case_config['DICESPLIT']))
ax.set_ylabel('percent of vessels above the Dice threshold',fontsize = 15)
ax.set_xlabel('Dice threshold',fontsize = 15)
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(case_config['DICERANGES'],dicePercentageLargeAdd,'o',color='r')
#plt.tick_params(labelsize=11)
ax.set_title('vessels larger than: '+str(case_config['DICESPLIT']))
ax.set_ylabel('percent of vessels above the Dice threshold',fontsize = 15)
ax.set_xlabel('Dice threshold',fontsize = 15)
plt.show()

'''
#plot number of vessels in dataset per diameter range
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(tablePerLength.loc[:,'x_uperBound'],tablePerLength.loc[:,'numVessels'],'o',color='r')
total=sum(tablePerLength.loc[:,'numVessels'])
percentage=[]
for count in range(tablePerLength.shape[0]):
    if total!=0:
        percentage.append((100.0/total)*tablePerLength.get_value(count,'numVessels'))
ax2 = ax.twinx()
ax2.plot(tablePerLength.loc[:,'x_uperBound'],percentage,'o',color='r')
ax2.set_ylabel('percent of all images in the dataset',fontsize = 15)
plt.tick_params(labelsize=11)
plt.show()

#plot ASSD per diameter range
figAssd,axAssd= plt.subplots(figsize=(15,15))
lAssd=axAssd.plot(tablePerLength.loc[:,'x_uperBound'],tablePerLength.loc[:,'ASSD'],color='y',linestyle='-')
aj=axAssd.plot(tablePerLength.loc[:,'x_uperBound'],np.array([assdaverage for i in xrange(tablePerLength.shape[0])]),color='y',linestyle='-.')
axAssd.set_title("Assd (s)"+case_config['RESULTS_DIR'])
plt.show()
plt.savefig(case_config['RESULTS_DIR']+'/Assd.png',dpi=600)
pickle.dump(axAssd, file(case_config['RESULTS_DIR']+'/Assd1.pickle', 'w'))

#plot HD per diameter range
figHd,axHd= plt.subplots(figsize=(15,15))
lHd=axHd.plot(tablePerLength.loc[:,'x_uperBound'],tablePerLength.loc[:,'HD'],color='b',linestyle='-')
aH=axHd.plot(tablePerLength.loc[:,'x_uperBound'],np.array([hdaverage for i in xrange(tablePerLength.shape[0])]),color='b',linestyle='-.')
axHd.set_title("Hausdorf (s)"+case_config['RESULTS_DIR'])
plt.savefig(case_config['RESULTS_DIR']+'/Hausdorf.png',dpi=600)
pickle.dump(axHd, file(case_config['RESULTS_DIR']+'/Hd1.pickle', 'w'))
plt.show()

#plot DICE per diameter range
figDc,axDc= plt.subplots(figsize=(15,15))
lDc=axDc.plot(tablePerLength.loc[:,'x_uperBound'],tablePerLength.loc[:,'DICE'],color='g',linestyle='-')
aDc=axDc.plot(tablePerLength.loc[:,'x_uperBound'],np.array([dcaverage for i in xrange(tablePerLength.shape[0])]),color='g',linestyle='-.')
axDc.set_title("Dice (l)"+case_config['RESULTS_DIR'])
plt.savefig(case_config['RESULTS_DIR']+'/Dice.png',dpi=600)
pickle.dump(axDc, file(case_config['RESULTS_DIR']+'/Dice1.pickle', 'w'))
plt.show()
'''
