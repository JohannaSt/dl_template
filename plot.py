import matplotlib.pyplot as plt
import csv
from csv import reader
import numpy as np
'''
#all Vessels I2INet Contour:
coronaryOnly=[100.0, 83.55899419729207, 56.86653771760155, 50.870406189555126, 49.32301740812379, 47.77562862669246, 45.261121856866545, 41.00580270793037, 35.78336557059962, 30.947775628626694, 0.21663442940038685]

pretrained= [100.0, 86.26692456479691, 57.44680851063829, 51.06382978723404, 50.29013539651837, 49.70986460348163, 48.16247582205029, 46.421663442940044, 43.90715667311412, 38.49129593810445, 0.22437137330754353]

general= [100.0, 94.97098646034817, 42.359767891682786, 26.4990328820116, 24.371373307543518, 22.82398452611218, 21.27659574468085, 19.535783365570598, 17.2147001934236, 13.539651837524177, 0.08704061895551257]


#small Vessels I2INet Contour:
coronaryOnly=np.genfromtxt('/home/johanna/dl_template/results/case1_coronaryOnly_useYc_WORC1/val/DiceThresholdPercentage.csv',delimiter=',')
pretrained=np.genfromtxt('/home/johanna/dl_template/results/case1_coronaryPretrained_useYc_WORC1/val/DiceThresholdPercentage.csv',delimiter=',')
general=np.genfromtxt('/home/johanna/dl_template/results/case1_loft_useYc_Contour_WORC1/val/DiceThresholdPercentage.csv',delimiter=',')

diceRange=[0.0,0.10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

plt.xlim([0.0,1.0])
plt.title('Dice threshold small vessels')
plt.plot(diceRange,coronaryOnly[1:,3],color='y',linestyle='-',label='coronary only')
plt.ylabel('percent of vessels above the Dice threshold',fontsize = 12)
plt.xlabel('Dice threshold',fontsize = 12)
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*coronaryOnly[1,2],color='y',linestyle='-.',label='mean coronary only:{}'.format(round(coronaryOnly[1,2],3)))
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,1]))*sum(coronaryOnly[1:,1])/len(coronaryOnly[1:,1]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,1])/len(coronaryOnly[1:,1]),2)))
plt.plot(diceRange,pretrained[1:,3],color='b',linestyle='-',label='coronary pretrained')
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*pretrained[1,2],color='b',linestyle='-.',label='mean coronary pretrained:{}'.format(round(pretrained[1,2],3)))
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,1]))*sum(pretrained[1:,1])/len(pretrained[1:,1]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,1])/len(pretrained[1:,1]),2)))
plt.plot(diceRange,general[1:,3],color='r',linestyle='-',label='general')
#plt.plot(general[1:,9],np.ones(len(general[1:,9]))*general[1,2],color='r',linestyle='-.',label='mean general:{}'.format(round(general[1,2],3)))
#plt.plot(general[1:,9],np.ones(len(general[1:,1]))*sum(general[1:,1])/len(general[1:,1]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,1])/len(general[1:,1]),2)))
plt.legend()
plt.savefig('./results/DiceThreshold_SmallVessels_useYc_WORC1')
plt.show()

'''
'''
coronaryOnly=np.genfromtxt('/home/johanna/dl_template/resultsFC/case1_coronaryOnlyTH0.3_useYc/val/metricPerDiameter.csv',delimiter=',')
#coronaryOnly1= np.genfromtxt('/home/johanna/dl_template/resultsFC/case1_coronaryOnlyTH0.3_useYc_useContour/val/metricPerDiameter.csv',delimiter=',')

coronaryOnly2=np.genfromtxt('/home/johanna/dl_template/resultsFC/case1_coronaryOnlyBoundaryTH0.3/val/metricPerDiameter.csv',delimiter=',')

pretrained=np.genfromtxt('./resultsFC/case1_coronaryPretrainedTH0.3Check/val/metricPerDiameter.csv',delimiter=',')
#pretrained1= np.genfromtxt('./resultsFC/case1_coronaryPretrained_TH0.3_useYc_useContour/val/metricPerDiameter.csv',delimiter=',')

pretrained2=np.genfromtxt('./resultsFC/case1_coronaryPretrainedTH0.3Boundary/val/metricPerDiameter.csv',delimiter=',')


general=np.genfromtxt('./resultsFC/case1_loft_useYcTestTH03/val/metricPerDiameter.csv',delimiter=',')
#general1= np.genfromtxt('./resultsFC/case1_loft_useYcTH0.3_useContour/val/metricPerDiameter.csv',delimiter=',')

general2=np.genfromtxt('./resultsFC/case1_loft_TH3Boundary/val/metricPerDiameter.csv',delimiter=',')

#plot assd
#fig, ax = plt.subplots(figsize=(20,20))
print 'Assd'
print coronaryOnly[1:,1]
print pretrained[1:,1]
print general[1:,1]

plt.xlim([0.0,1.5])
plt.title('Assd(s)')
plt.ylabel('Assd',fontsize = 12)
plt.xlabel('vessel size',fontsize = 12)
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,1],color='y',linestyle='-',label='coronary only useYc')
plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*coronaryOnly[1,2],color='y',linestyle='-.',label='mean coronary only useYc: {}'.format(round(coronaryOnly[1,2],3)))

plt.plot(coronaryOnly2[1:,9],coronaryOnly2[1:,1],color='g',linestyle='-',label='coronary only')
plt.plot(coronaryOnly2[1:,9],np.ones(len(coronaryOnly2[1:,9]))*coronaryOnly2[1,2],color='g',linestyle='-.',label='mean coronary only: {}'.format(round(coronaryOnly2[1,2],3)))

#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,1]))*sum(coronaryOnly[1:,1])/len(coronaryOnly[1:,1]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,1])/len(coronaryOnly[1:,1]),2)))
plt.plot(pretrained[1:,9],pretrained[1:,1],color='b',linestyle='-',label='coronary pretrained useYc')
plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*pretrained[1,2],color='b',linestyle='-.',label='mean coronary pretrained useYc: {}'.format(round(pretrained[1,2],3)))

plt.plot(pretrained2[1:,9],pretrained2[1:,1],color='k',linestyle='-',label='coronary pretrained')
plt.plot(pretrained2[1:,9],np.ones(len(pretrained2[1:,9]))*pretrained2[1,2],color='k',linestyle='-.',label='mean coronary pretrained: {}'.format(round(pretrained2[1,2],3)))

#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,1]))*sum(pretrained[1:,1])/len(pretrained[1:,1]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,1])/len(pretrained[1:,1]),2)))
plt.plot(general[1:,9],general[1:,1],color='r',linestyle='-',label='general useYc')
plt.plot(general[1:,9],np.ones(len(general[1:,9]))*general[1,2],color='r',linestyle='-.',label='mean general useYc: {}'.format(round(general[1,2],3)))

plt.plot(general2[1:,9],general2[1:,1],color='m',linestyle='-',label='general')
plt.plot(general2[1:,9],np.ones(len(general2[1:,9]))*general2[1,2],color='m',linestyle='-.',label='mean general: {}'.format(round(general2[1,2],3)))
#plt.plot(general[1:,9],np.ones(len(general[1:,1]))*sum(general[1:,1])/len(general[1:,1]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,1])/len(general[1:,1]),2)))

plt.legend(bbox_to_anchor=(1.04,0), loc="lower left")
plt.savefig('./resultsFC/Test',bbox_inches="tight")
plt.show()


#plot Hausdorf
#fig, ax = plt.subplots(figsize=(20,20))
print 'Hausdorf'
print coronaryOnly[1:,5]
print pretrained[1:,5]
print general[1:,5]
plt.xlim([0.0,1.5])
plt.title('Hausdorf(s)')
plt.ylabel('Hausdorf',fontsize = 12)
plt.xlabel('vessel size',fontsize = 12)
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,5],color='y',linestyle='-',label='coronary only useYc')
plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*(coronaryOnly[1,6]),color='y',linestyle='-.',label='mean coronary only useYc:{}'.format(round(coronaryOnly[1,6],3)))

plt.plot(coronaryOnly2[1:,9],coronaryOnly2[1:,5],color='g',linestyle='-',label='coronary only')
plt.plot(coronaryOnly2[1:,9],np.ones(len(coronaryOnly2[1:,9]))*(coronaryOnly2[1,6]),color='g',linestyle='-.',label='mean coronary only:{}'.format(round(coronaryOnly2[1,6],3)))

#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,5]))*sum(coronaryOnly[1:,5])/len(coronaryOnly[1:,5]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,5])/len(coronaryOnly[1:,5]),2)))
plt.plot(pretrained[1:,9],pretrained[1:,5],color='b',linestyle='-',label='coronary pretrained useYc')
plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*(pretrained[1,6]),color='b',linestyle='-.',label='mean coronary pretrained useYc:{}'.format(round(pretrained[1,6],3)))

plt.plot(pretrained2[1:,9],pretrained2[1:,5],color='k',linestyle='-',label='coronary pretrained')
plt.plot(pretrained2[1:,9],np.ones(len(pretrained2[1:,9]))*(pretrained2[1,6]),color='k',linestyle='-.',label='mean coronary pretrained:{}'.format(round(pretrained2[1,6],3)))

#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,5]))*sum(pretrained[1:,5])/len(pretrained[1:,5]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,5])/len(pretrained[1:,5]),2)))
plt.plot(general[1:,9],general[1:,5],color='r',linestyle='-',label='general useYc')
plt.plot(general[1:,9],np.ones(len(general[1:,9]))*(general[1,6]),color='r',linestyle='-.',label='mean general useYc:{}'.format(round(general[1,6],3)))

plt.plot(general2[1:,9],general2[1:,5],color='m',linestyle='-',label='general')
plt.plot(general2[1:,9],np.ones(len(general2[1:,9]))*(general2[1,6]),color='m',linestyle='-.',label='mean general:{}'.format(round(general2[1,6],3)))
#plt.plot(general[1:,9],np.ones(len(general[1:,5]))*sum(general[1:,5])/len(general[1:,5]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,5])/len(general[1:,5]),2)))
plt.legend(bbox_to_anchor=(1.04,0), loc="lower left")
plt.savefig('./resultsFC/HausdorfTest',bbox_inches="tight")
plt.show()

#plot Dice
#fig, ax = plt.subplots(figsize=(20,20))
print 'Dice'
print coronaryOnly[1:,3]
print pretrained[1:,3]
print general[1:,3]
plt.xlim([0.0,1.5])
plt.title('Dice(l)')
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,3],color='y',linestyle='-',label='coronary only useYc')
plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,9]))*coronaryOnly[1,4],color='y',linestyle='-.',label='mean coronary only useYc:{}'.format(round(coronaryOnly[1,4],3)))

plt.plot(coronaryOnly2[1:,9],coronaryOnly2[1:,3],color='g',linestyle='-',label='coronary only')
plt.plot(coronaryOnly2[1:,9],np.ones(len(coronaryOnly2[1:,9]))*coronaryOnly2[1,4],color='g',linestyle='-.',label='mean coronary only:{}'.format(round(coronaryOnly2[1,4],3)))
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,3]))*sum(coronaryOnly[1:,3])/len(coronaryOnly[1:,3]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,3])/len(coronaryOnly[1:,3]),2)))
plt.plot(pretrained[1:,9],pretrained[1:,3],color='b',linestyle='-',label='coronary pretrained useYc')
plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,9]))*pretrained[1,4],color='b',linestyle='-.',label='mean coronary pretrained useYc:{}'.format(round(pretrained[1,4],3)))

plt.plot(pretrained2[1:,9],pretrained2[1:,3],color='k',linestyle='-',label='coronary pretrained')
plt.plot(pretrained2[1:,9],np.ones(len(pretrained2[1:,9]))*pretrained2[1,4],color='k',linestyle='-.',label='mean coronary pretrained:{}'.format(round(pretrained2[1,4],3)))
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,3]))*sum(pretrained[1:,3])/len(pretrained[1:,3]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,3])/len(pretrained[1:,3]),2)))
plt.plot(general[1:,9],general[1:,3],color='r',linestyle='-',label='general useYc')
plt.plot(general[1:,9],np.ones(len(general[1:,9]))*general[1,4],color='r',linestyle='-.',label='mean general useYc:{}'.format(round(general[1,4],3)))

plt.plot(general2[1:,9],general2[1:,3],color='m',linestyle='-',label='general')
plt.plot(general2[1:,9],np.ones(len(general2[1:,9]))*general2[1,4],color='m',linestyle='-.',label='mean general:{}'.format(round(general2[1,4],3)))
#plt.plot(general[1:,9],np.ones(len(general[1:,3]))*sum(general[1:,3])/len(general[1:,3]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,3])/len(general[1:,3]),2)))
plt.legend(bbox_to_anchor=(1.04,0), loc="lower left")
plt.savefig('./resultsFC/DiceTEST',bbox_inches="tight")
plt.show()
'''
'''
#plot Vessel diameter
#fig, ax = plt.subplots(figsize=(20,20))
print 'number of Vessels'
print coronaryOnly[1:,7]
#print pretrained[1:,7]
plt.xlim([0.0,1.5])
plt.title('number of Vessels per diameter range(validation set)')
plt.plot(coronaryOnly[1:,9],coronaryOnly[1:,7],color='r',linestyle='-',label='coronary')
#plt.plot(coronaryOnly[1:,9],np.ones(len(coronaryOnly[1:,7]))*(coronaryOnly[1:,7])/len(coronaryOnly[1:,7]),color='y',linestyle='-.',label='mean coronary only:{}'.format(round(sum(coronaryOnly[1:,7])/len(coronaryOnly[1:,7]),2)))
#plt.plot(pretrained[1:,9],pretrained[1:,7],color='b',linestyle='-',label='coronary pretrained')
#plt.plot(pretrained[1:,9],np.ones(len(pretrained[1:,7]))*(pretrained[1:,7])/len(pretrained[1:,7]),color='b',linestyle='-.',label='mean pretrained:{}'.format(round(sum(pretrained[1:,7])/len(pretrained[1:,7]),2)))
#plt.scatter(general[1:,9],general[1:,7],color='r',linestyle='-',label='general')
#plt.plot(general[1:,9],np.ones(len(general[1:,7]))*(general[1:,7])/len(general[1:,7]),color='r',linestyle='-.',label='mean general:{}'.format(round(sum(general[1:,7])/len(general[1:,7]),2)))
#plt.legend()
plt.savefig('./resultsFC/vesselsPerRangeValidationCoronaryLine')
plt.show()
'''
#plot Vessel diameter in training datasets
coronaryTrain=np.genfromtxt('./resultsFC/case1_coronaryOnlyTrainData/val/metricPerDiameter.csv',delimiter=',')
#'./results/newDice/case1_coronaryOnly_useYcTH0.3_SmallVessels/val/metricPerDiameter.csv',delimiter=',')
generalTrain=np.genfromtxt('./resultsFC/case1_loft_trainData/val/metricPerDiameter.csv',delimiter=',')
#'./resultsFC/case1_loft_trainDataSmall/val/metricPerDiameter.csv',delimiter=',')
print coronaryTrain[1:16,9]
print coronaryTrain[1:16,7]
#print pretrained[1:,4]
print generalTrain[1:16,7]
plt.ylabel('percentage of vessels [%]',fontsize = 12)
plt.xlabel('vessel size [cm]',fontsize = 12)
plt.xlim([0.06,1.6])
plt.title('percentage of vessels in the dataset per radius range (training set)')
plt.plot(coronaryTrain[1:16,9],coronaryTrain[1:16,10],'o',color='y',label='coronary')
plt.plot(1.55,coronaryTrain[16,10],'o',color='y')
#coronaryTrain[-1,7],'o',color='y')
plt.plot(generalTrain[1:16,9],generalTrain[1:16,11],'o',color='r',label='general')
plt.plot(1.55,100/14747*119,'o',color='r')
plt.legend()
plt.savefig('./results/newDice/vesselsPerRangeTrainingPercentage')
plt.show()

'''
coronaryTrain=np.genfromtxt('./results/newDice/case1_coronaryOnly_useYcTH0.3_SmallVessels/val/metricPerDiameter.csv',delimiter=',')
#'./resultsFC/case1_coronaryOnlyValData/val/metricPerDiameter.csv',delimiter=',')
#fig, ax = plt.subplots(figsize=(10,10))
plt.title('number of vessels in the dataset per radius range (validation set)')
plt.xlim([0.06,0.27])
plt.plot(coronaryTrain[1:9,9],coronaryTrain[1:9,7],'o',color='y',label='coronary')
plt.plot(0.26,coronaryTrain[9,7],'o',color='y')
#plt.plot(0.255,coronaryTrain[9,7],'o',color='y')
plt.xlabel('vessel size [cm]',fontsize = 12)
plt.ylabel('number of vessels',fontsize = 12)
ax2 = plt.twinx()
ax2.plot(coronaryTrain[1:9,9],coronaryTrain[1:9,10],'o',color='y',label='coronary')
ax2.plot(0.26,coronaryTrain[9,10],'o',color='y')
#ax2.plot(0.255,coronaryTrain[9,10],'o',color='y')
ax2.set_ylabel('percentage of all images in the dataset',fontsize = 12)
plt.tick_params(labelsize=12)
plt.savefig('./results/newDice/vesselsPerRangeValidation_Coronary')
plt.show()
'''
