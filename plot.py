import matplotlib.pyplot as plt
import csv
from csv import reader
import numpy as np
coronaryOnly=np.genfromtxt('./results/case1_coronaryOnly_useYcTest/val/metricPerDiameter.csv',delimiter=',')

pretrained= np.genfromtxt('./results/case1_coronaryOnlyCheck2_useYc/val/metricPerDiameter.csv',delimiter=',')

general=np.genfromtxt('./results/case1_coronaryOnlyCheck_useYc/val/metricPerDiameter.csv',delimiter=',')

'''
coronaryOnly=np.genfromtxt('./results/case1_coronaryOnly_useYcTest/val/metricPerDiameter.csv',delimiter=',')

pretrained= np.genfromtxt('./results/case1_coronary_useYc_newMean/val/metricPerDiameter.csv',delimiter=',')

general=np.genfromtxt('./results/case1_loft_useYc_valCoronary/val/metricPerDiameter.csv',delimiter=',')
'''
#plot assd
#fig, ax = plt.subplots(figsize=(20,20))
print 'Assd'
print coronaryOnly[1:,1]
print pretrained[1:,1]
print general[1:,1]

plt.xlim([0.0,1.5])
plt.title('Assd(s)')
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,1],color='y',linestyle='-',label='coronary only')
plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*coronaryOnly[1,2],color='y',linestyle='-.',label='mean coronary only:{}'.format(round(coronaryOnly[1,2],3)))
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,1]))*sum(coronaryOnly[1:,1])/len(coronaryOnly[1:,1]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,1])/len(coronaryOnly[1:,1]),2)))
plt.plot(pretrained[1:,9],pretrained[1:,1],color='b',linestyle='-',label='coronary pretrained')
plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*pretrained[1,2],color='b',linestyle='-.',label='mean coronary pretrained:{}'.format(round(pretrained[1,2],3)))
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,1]))*sum(pretrained[1:,1])/len(pretrained[1:,1]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,1])/len(pretrained[1:,1]),2)))
plt.plot(general[1:,9],general[1:,1],color='r',linestyle='-',label='general')
plt.plot(general[1:,9],np.ones(len(general[1:,9]))*general[1,2],color='r',linestyle='-.',label='mean general:{}'.format(round(general[1,2],3)))
#plt.plot(general[1:,9],np.ones(len(general[1:,1]))*sum(general[1:,1])/len(general[1:,1]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,1])/len(general[1:,1]),2)))
plt.legend()
plt.savefig('./results/AssdCoronary')
plt.show()


#plot Hausdorf
#fig, ax = plt.subplots(figsize=(20,20))
print 'Hausdorf'
print coronaryOnly[1:,5]
print pretrained[1:,5]
print general[1:,5]
plt.xlim([0.0,1.5])
plt.title('Hausdorf(s)')
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,5],color='y',linestyle='-',label='coronary only')
plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*(coronaryOnly[1,6]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(coronaryOnly[1,6],3)))
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,5]))*sum(coronaryOnly[1:,5])/len(coronaryOnly[1:,5]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,5])/len(coronaryOnly[1:,5]),2)))
plt.plot(pretrained[1:,9],pretrained[1:,5],color='b',linestyle='-',label='coronary pretrained')
plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*(pretrained[1,6]),color='b',linestyle='-.',label='mean coronary pretrained:{}'.format(round(pretrained[1,6],3)))

#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,5]))*sum(pretrained[1:,5])/len(pretrained[1:,5]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,5])/len(pretrained[1:,5]),2)))
plt.plot(general[1:,9],general[1:,5],color='r',linestyle='-',label='general')
plt.plot(general[1:,9],np.ones(len(general[1:,9]))*(general[1,6]),color='r',linestyle='-.',label='mean general:{}'.format(round(general[1,6],3)))

#plt.plot(general[1:,9],np.ones(len(general[1:,5]))*sum(general[1:,5])/len(general[1:,5]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,5])/len(general[1:,5]),2)))
plt.legend()
plt.savefig('./results/HausdorfCoronary')
plt.show()

#plot Dice
#fig, ax = plt.subplots(figsize=(20,20))
print 'Dice'
print coronaryOnly[1:,3]
print pretrained[1:,3]
print general[1:,3]
plt.xlim([0.0,1.5])
plt.title('Dice(l)')
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,3],color='y',linestyle='-',label='coronary only')
plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*coronaryOnly[1,4],color='y',linestyle='-.',label='mean coronary only:{}'.format(round(coronaryOnly[1,4],3)))
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,3]))*sum(coronaryOnly[1:,3])/len(coronaryOnly[1:,3]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,3])/len(coronaryOnly[1:,3]),2)))
plt.plot(pretrained[1:,9],pretrained[1:,3],color='b',linestyle='-',label='coronary pretrained')
plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*pretrained[1,4],color='b',linestyle='-.',label='mean coronary pretrained:{}'.format(round(pretrained[1,4],3)))
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,3]))*sum(pretrained[1:,3])/len(pretrained[1:,3]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,3])/len(pretrained[1:,3]),2)))
plt.plot(general[1:,9],general[1:,3],color='r',linestyle='-',label='general')
plt.plot(general[1:,9],np.ones(len(general[1:,9]))*general[1,4],color='r',linestyle='-.',label='mean general:{}'.format(round(general[1,4],3)))
#plt.plot(general[1:,9],np.ones(len(general[1:,3]))*sum(general[1:,3])/len(general[1:,3]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,3])/len(general[1:,3]),2)))
plt.legend()
plt.savefig('./results/DiceCoronary')
plt.show()
'''
#plot Vessel diameter
#fig, ax = plt.subplots(figsize=(20,20))
print 'number of Vessels'
print coronaryOnly[1:,7]
#print pretrained[1:,7]
print general[1:,7]
plt.xlim([0.0,2.0])
plt.title('number of Vessels per diameter range(validation set)')
plt.scatter(coronaryOnly[1:,9],coronaryOnly[1:,7],color='y',linestyle='-',label='coronary')
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,7]))*(coronaryOnly[1:,7])/len(coronaryOnly[1:,7]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,7])/len(coronaryOnly[1:,7]),2)))
#plt.plot(pretrained[1:,9],pretrained[1:,7],color='b',linestyle='-',label='coronary pretrained')
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,7]))*(pretrained[1:,7])/len(pretrained[1:,7]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,7])/len(pretrained[1:,7]),2)))
plt.scatter(general[1:,9],general[1:,7],color='r',linestyle='-',label='general')
#plt.plot(general[1:,9],np.ones(len(general[1:,7]))*(general[1:,7])/len(general[1:,7]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,7])/len(general[1:,7]),2)))
plt.legend()
plt.savefig('./results/vesselsPerRangeValidation')
plt.show()

#plot Vessel diameter in training datasets
coronaryTrain=np.genfromtxt('./results/caseTrainingDataDiameters/val/metricPerDiameter.csv',delimiter=',')
generalTrain=np.genfromtxt('./results/caseTrainingDatasetGeneral/val/metricPerDiameter.csv',delimiter=',')
print 'number of Vessels'
print coronaryTrain[1:,4]
#print pretrained[1:,4]
print generalTrain[1:,4]
plt.xlim([0.0,2.0])
plt.title('number of Vessels per diameter range(training set)')
plt.scatter(coronaryTrain[1:,9],coronaryTrain[1:,4],color='y',linestyle='-',label='coronary')
plt.scatter(generalTrain[1:,9],generalTrain[1:,4],color='r',linestyle='-',label='general')
plt.legend()
plt.savefig('./results/vesselsPerRangeTraining')
plt.show()
'''
