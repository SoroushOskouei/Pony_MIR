''' K-fold and MonteCarlo cross-validation 
Made for UEF BBC group project 06.06.2022
by Soroush Oskouei
'''

import pandas as pd
# !pip install git+https://github.com/uef-bbc/nippy
import nippy
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics

print('input file name:')
name=input()

dataframe = pd.read_csv('name',header=None)
Names=dataframe[0].values.tolist()
wavelength = pd.DataFrame([*range(1,523,1)])

spectral = (dataframe.T)[4:523] # Rows = wavelength, Columns = samples
pipelines = nippy.read_configuration('noder.ini')
datasets = nippy.nippy(wavelength, spectral, pipelines)

for NIPPYNUM in range(len(datasets)):

  # print('Which Nippy dataset should I use? (starting from 0)')
  method='MC'
  # NIPPYNUM=int(input())

  print('NIPPYNUM=', NIPPYNUM)
  spectra=pd.DataFrame(datasets[NIPPYNUM][1].T)
  DDs = (dataframe.T)[3:4]


  X=[]
  Y=[]
  ponies=[]

  for i in Names:
  # unhealthy
      if i[4:9]=='02LR2' or i[4:9]=='03LR2' or i[4:9]=='04RR2' or i[4:9]=='05RR2' or i[4:9]=='06Li4' or i[4:9]=='07Li4' or i[4:9]=='08Ri4' or i[4:9]=='09Ri4' or i[4:9]=='10Ri4' or i[4:9]=='02Li4' or i[4:9]=='03Li4' or i[4:9]=='04Ri4' or i[4:9]=='05Ri4' or i[4:9]=='06LR2' or i[4:9]=='07LR2' or i[4:9]=='08RR2' or i[4:9]=='09RR2' or i[4:9]=='10RR2':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])
  # healthy
      if i[4:9]=='02RR2' or i[4:9]=='03RR2' or i[4:9]=='04LR2' or i[4:9]=='05LR2' or i[4:9]=='06Ri4' or i[4:9]=='07Ri4' or i[4:9]=='08Li4' or i[4:9]=='09Li4' or i[4:9]=='10Li4' or i[4:9]=='02Ri4' or i[4:9]=='03Ri4' or i[4:9]=='04Li4' or i[4:9]=='05Li4' or i[4:9]=='06RR2' or i[4:9]=='07RR2' or i[4:9]=='08LR2' or i[4:9]=='09LR2' or i[4:9]=='10LR2':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])


  # unhealthy Kissing
      if i[4:9]=='02LR1' or i[4:9]=='03LR1' or i[4:9]=='04RR1' or i[4:9]=='05RR1' or i[4:9]=='06Li3' or i[4:9]=='07Li3' or i[4:9]=='08Ri3' or i[4:9]=='09Ri3' or i[4:9]=='10Ri3' or i[4:9]=='02Li3' or i[4:9]=='03Li3' or i[4:9]=='04Ri3' or i[4:9]=='05Ri3' or i[4:9]=='06LR1' or i[4:9]=='07LR1' or i[4:9]=='08RR1' or i[4:9]=='09RR1' or i[4:9]=='10RR1':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])
  # healthy Kissing
      if i[4:9]=='02RR1' or i[4:9]=='03RR1' or i[4:9]=='04LR1' or i[4:9]=='05LR1' or i[4:9]=='06Ri3' or i[4:9]=='07Ri3' or i[4:9]=='08Li3' or i[4:9]=='09Li3' or i[4:9]=='10Li3' or i[4:9]=='02Ri3' or i[4:9]=='03Ri3' or i[4:9]=='04Li3' or i[4:9]=='05Li3' or i[4:9]=='06RR1' or i[4:9]=='07RR1' or i[4:9]=='08LR1' or i[4:9]=='09LR1' or i[4:9]=='10LR1':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])


  # Blunt
      if i[4:9]=='02LR2' or i[4:9]=='03LR2' or i[4:9]=='04RR2' or i[4:9]=='05RR2' or i[4:9]=='06Li4' or i[4:9]=='07Li4' or i[4:9]=='08Ri4' or i[4:9]=='09Ri4' or i[4:9]=='10Ri4':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])
  # Sharp
      if i[4:9]=='02Li4' or i[4:9]=='03Li4' or i[4:9]=='04Ri4' or i[4:9]=='05Ri4' or i[4:9]=='06LR2' or i[4:9]=='07LR2' or i[4:9]=='08RR2' or i[4:9]=='09RR2' or i[4:9]=='10RR2':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])

  # Blunt Kissing
      if i[4:9]=='02LR1' or i[4:9]=='03LR1' or i[4:9]=='04RR1' or i[4:9]=='05RR1' or i[4:9]=='06Li3' or i[4:9]=='07Li3' or i[4:9]=='08Ri3' or i[4:9]=='09Ri3' or i[4:9]=='10Ri3':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])
  # Sharp Kissing
      if i[4:9]=='02Li3' or i[4:9]=='03Li3' or i[4:9]=='04Ri3' or i[4:9]=='05Ri3' or i[4:9]=='06LR1' or i[4:9]=='07LR1' or i[4:9]=='08RR1' or i[4:9]=='09RR1' or i[4:9]=='10RR1':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])

  # Grooved
      if i[4:9]=='02LR2' or i[4:9]=='03LR2' or i[4:9]=='04RR2' or i[4:9]=='05RR2' or i[4:9]=='06Li4' or i[4:9]=='07Li4' or i[4:9]=='08Ri4' or i[4:9]=='09Ri4' or i[4:9]=='10Ri4' or i[4:9]=='02Li4' or i[4:9]=='03Li4' or i[4:9]=='04Ri4' or i[4:9]=='05Ri4' or i[4:9]=='06LR2' or i[4:9]=='07LR2' or i[4:9]=='08RR2' or i[4:9]=='09RR2' or i[4:9]=='10RR2':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])
  # Grooved Kissing
      if i[4:9]=='02LR1' or i[4:9]=='03LR1' or i[4:9]=='04RR1' or i[4:9]=='05RR1' or i[4:9]=='06Li3' or i[4:9]=='07Li3' or i[4:9]=='08Ri3' or i[4:9]=='09Ri3' or i[4:9]=='10Ri3' or i[4:9]=='02Li3' or i[4:9]=='03Li3' or i[4:9]=='04Ri3' or i[4:9]=='05Ri3' or i[4:9]=='06LR1' or i[4:9]=='07LR1' or i[4:9]=='08RR1' or i[4:9]=='09RR1' or i[4:9]=='10RR1':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(DDs[Names.index(i)].values.tolist())
          ponies.append(i[1:3])

  accuracies=[]
  import random
  import statistics

  # print('How many pony groups for test? ')
  HowmanyPony=1
  # HowmanyPony=int(input())
  mses=[]
  overallTrue=[]
  overallPred=[]
  if method=='MC':
    for k in range(10):
      # for j in [30, 33, 35, 37, 40, 41, 42, 47, 64]:
        X_train=[]
        X_test=[]
        Y_train=[]
        Y_test=[]
        
        for i in range(len(ponies)):
          X_train.append(X[i])
          Y_train.append(Y[i])

        # temp=random.randint(0,len(X_train)-1)
        pony_copy=ponies.copy()

        for rnd in range(HowmanyPony):
          temp=random.randint(0,len(X_train)-1)

          if temp<len(X_train)-1 and ponies[temp]==ponies[temp+1]:
            X_test.append(X_train[temp])
            X_test.append(X_train[temp+1])
            Y_test.append(Y_train[temp])
            Y_test.append(Y_train[temp+1])
            # X_train.pop(temp)
            # X_train.pop(temp)
            # Y_train.pop(temp)
            # Y_train.pop(temp)   
          else:
            X_test.append(X_train[temp])
            X_test.append(X_train[temp-1])
            Y_test.append(Y_train[temp])
            Y_test.append(Y_train[temp-1])
            # X_train.pop(temp)
            # X_train.pop(temp-1)
            # Y_train.pop(temp)
            # Y_train.pop(temp-1)
      #   print(len(X_test))
        # pony_copy=ponies.copy()
        
          for l in pony_copy:
            if l==ponies[temp]:
              pony_copy.pop(pony_copy.index(l))
              X_train.pop(pony_copy.index(l))
              Y_train.pop(pony_copy.index(l))
        # print('test:', Y_test)
        # reg=LinearRegression().fit(X_train, Y_train)
        reg=RandomForestRegressor(max_depth=8, random_state=300).fit(X_train, Y_train)
        # print(X_test)
        # print(Y_test)
        Y_pred=reg.predict(X_test)
        for kk in range(len(X_test)):
          overallPred.append(Y_pred[kk])
          overallTrue.append(Y_test[kk])
        from sklearn.metrics import mean_squared_error
        mses.append(mean_squared_error(Y_test, Y_pred))
  # Sharps_V_B4
  # print(sum(mses)/len(mses))
  import statistics
  mses.sort()
  # print(statistics.median(mses))
  print('MSEP:')
  print(0.007/(max(np.array(Y_train))-min(np.array(Y_train)))*100)

  import matplotlib.pyplot as plt
  plt.scatter(overallPred, overallTrue, c="b", alpha=0.8)
  plt.xlabel("Predictions")
  plt.ylabel("True Values")
  plt.show()

  #  1 -> thickness
  #  2 -> Equilibrium modulus
  #  3 -> instant modulus

