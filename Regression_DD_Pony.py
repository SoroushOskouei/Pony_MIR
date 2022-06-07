''' Regression models made for pony DD values 
Made for UEF BBC group project 17.24.2022
by Soroush Oskouei
'''

import pandas as pd
import numpy as np
import nippy
from sklearn.linear_model import LinearRegression
import numpy as np

print('File name:')
name='Blunts_V_B2.csv'


print('0-5 or 0-10')
# zone=input()
zone='0-10'
print(zone)
dataframe = pd.read_csv('Spectra/'+name,header=None)

Names=dataframe[1].values.tolist()
wavelength = pd.DataFrame([*range(1,101,1)])
spectral = (dataframe.T)[2:101] # Rows = wavelength, Columns = samples
pipelines = nippy.read_configuration('noder.ini')
datasets = nippy.nippy(wavelength, spectral, pipelines)

for NIPPYNUM in range(len(datasets)):
  print('NIPPYNUM=', NIPPYNUM)
  spectra=pd.DataFrame(datasets[NIPPYNUM][1])

  DDs = pd.read_csv('DD/'+name)

  from random import randint

  mses=[]
  overallTrue=[]
  overallPred=[]
  for k in range(20):

    TestPonyNumIndex=randint(0,len(dataframe[1])-1)
    TestPonyNum=dataframe[1][TestPonyNumIndex]


    from numpy.ma.core import append
    X_test=[]
    X_train=[]
    theList_train=[]
    theList_test=[]
    for i in range(len(dataframe[1])):
      if dataframe[1][i]!=TestPonyNum:
        X_train.append(spectra[i].values.tolist())
        theList_train.append([dataframe[1][i], dataframe[0][i]])
      else:
        X_test.append(spectra[i].values.tolist())
        theList_test.append([dataframe[1][i], dataframe[0][i]])

    temp=randint(0,len(X_test)-1)
    X_test.pop(temp)
    temp=randint(0,len(X_test)-1)
    X_test.pop(temp)
    y_train=[]
    y_test=[]
    for i in range(len(X_train)):
      for j in range(len(DDs)):
        if theList_train[i]==[DDs['Pony'][j], DDs['ROI'][j]]:
          # print([DDs['Pony'][j], DDs['ROI'][j]])
          y_train.append(DDs[zone][j])

    for i in range(len(X_test)):
      for j in range(len(DDs)):
        if theList_test[i]==[DDs['Pony'][j], DDs['ROI'][j]]:
          # print([DDs['Pony'][j], DDs['ROI'][j]])
          y_test.append(DDs[zone][j])


    reg=LinearRegression().fit(X_train, y_train)
    y_pred=reg.predict(X_test)
    for kk in range(len(X_test)):
      overallPred.append(y_pred[kk])
      overallTrue.append(y_test[kk])
    from sklearn.metrics import mean_squared_error
    mses.append(mean_squared_error(y_test, y_pred))

  # print(sum(mses)/len(mses))
  import statistics
  mses.sort()
  print(statistics.median(mses))
  import matplotlib.pyplot as plt
  plt.scatter(overallPred, overallTrue, c="b", alpha=0.8)
  plt.xlabel("Predictions")
  plt.ylabel("True Values")
  plt.show()

