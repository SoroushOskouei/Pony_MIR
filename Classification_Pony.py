''' K-fold and MonteCarlo cross-validation 
Made for UEF BBC group project 10.05.2022
by Soroush Oskouei
'''

import pandas as pd
''' The next line is used for installation of nippy '''

# !pip install git+https://github.com/uef-bbc/nippy
import nippy

from sklearn import svm
import pickle
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
SVM_clf=svm.SVC(gamma='scale', decision_function_shape='ovo')

''' Loading the data '''

print('Type the name of the CSV data to be used:')
name=input()

dataframe = pd.read_csv(name,header=None)
Names=dataframe[0].values.tolist()
wavelength = pd.DataFrame([*range(1,523,1)])

spectral = (dataframe.T)[4:523] # Rows = wavelength, Columns = samples
pipelines = nippy.read_configuration('noder.ini')
datasets = nippy.nippy(wavelength, spectral, pipelines)

for NIPPYNUM in range(len(datasets)):

  # print('Which Nippy dataset should I use? (starting from 0)')
  # method='MC'
  # NIPPYNUM=int(input())
  
  ''' Selecting the preprocessed data '''

  print('NIPPYNUM=', NIPPYNUM)
  spectra=pd.DataFrame(datasets[NIPPYNUM][1].T)

  # choose the method here:
  print('Type the method (K-fold  or  MC):  -> MC')
  method='MC'




  X=[]
  Y=[]
  ponies=[]
  
  ''' Select the wanted groups for analysis '''

  for i in Names:
  # unhealthy
      if i[4:9]=='02LR2' or i[4:9]=='03LR2' or i[4:9]=='04RR2' or i[4:9]=='05RR2' or i[4:9]=='06Li4' or i[4:9]=='07Li4' or i[4:9]=='08Ri4' or i[4:9]=='09Ri4' or i[4:9]=='10Ri4' or i[4:9]=='02Li4' or i[4:9]=='03Li4' or i[4:9]=='04Ri4' or i[4:9]=='05Ri4' or i[4:9]=='06LR2' or i[4:9]=='07LR2' or i[4:9]=='08RR2' or i[4:9]=='09RR2' or i[4:9]=='10RR2':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(0)
          ponies.append(i[1:3])
  # healthy
      if i[4:9]=='02RR2' or i[4:9]=='03RR2' or i[4:9]=='04LR2' or i[4:9]=='05LR2' or i[4:9]=='06Ri4' or i[4:9]=='07Ri4' or i[4:9]=='08Li4' or i[4:9]=='09Li4' or i[4:9]=='10Li4' or i[4:9]=='02Ri4' or i[4:9]=='03Ri4' or i[4:9]=='04Li4' or i[4:9]=='05Li4' or i[4:9]=='06RR2' or i[4:9]=='07RR2' or i[4:9]=='08LR2' or i[4:9]=='09LR2' or i[4:9]=='10LR2':
          X.append(spectra[Names.index(i)].values.tolist())
          Y.append(1)
          ponies.append(i[1:3])


  # # unhealthy Kissing
  #     if i[4:9]=='02LR1' or i[4:9]=='03LR1' or i[4:9]=='04RR1' or i[4:9]=='05RR1' or i[4:9]=='06Li3' or i[4:9]=='07Li3' or i[4:9]=='08Ri3' or i[4:9]=='09Ri3' or i[4:9]=='10Ri3' or i[4:9]=='02Li3' or i[4:9]=='03Li3' or i[4:9]=='04Ri3' or i[4:9]=='05Ri3' or i[4:9]=='06LR1' or i[4:9]=='07LR1' or i[4:9]=='08RR1' or i[4:9]=='09RR1' or i[4:9]=='10RR1':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(0)
  #         ponies.append(i[1:3])
  # # healthy Kissing
  #     if i[4:9]=='02RR1' or i[4:9]=='03RR1' or i[4:9]=='04LR1' or i[4:9]=='05LR1' or i[4:9]=='06Ri3' or i[4:9]=='07Ri3' or i[4:9]=='08Li3' or i[4:9]=='09Li3' or i[4:9]=='10Li3' or i[4:9]=='02Ri3' or i[4:9]=='03Ri3' or i[4:9]=='04Li3' or i[4:9]=='05Li3' or i[4:9]=='06RR1' or i[4:9]=='07RR1' or i[4:9]=='08LR1' or i[4:9]=='09LR1' or i[4:9]=='10LR1':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(1)
  #         ponies.append(i[1:3])


  # # Blunt
  #     if i[4:9]=='02LR2' or i[4:9]=='03LR2' or i[4:9]=='04RR2' or i[4:9]=='05RR2' or i[4:9]=='06Li4' or i[4:9]=='07Li4' or i[4:9]=='08Ri4' or i[4:9]=='09Ri4' or i[4:9]=='10Ri4':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(0)
  #         ponies.append(i[1:3])
  # # Sharp
  #     if i[4:9]=='02Li4' or i[4:9]=='03Li4' or i[4:9]=='04Ri4' or i[4:9]=='05Ri4' or i[4:9]=='06LR2' or i[4:9]=='07LR2' or i[4:9]=='08RR2' or i[4:9]=='09RR2' or i[4:9]=='10RR2':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(1)
  #         ponies.append(i[1:3])

  # # Blunt Kissing
  #     if i[4:9]=='02LR1' or i[4:9]=='03LR1' or i[4:9]=='04RR1' or i[4:9]=='05RR1' or i[4:9]=='06Li3' or i[4:9]=='07Li3' or i[4:9]=='08Ri3' or i[4:9]=='09Ri3' or i[4:9]=='10Ri3':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(0)
  #         ponies.append(i[1:3])
  # # Sharp Kissing
  #     if i[4:9]=='02Li3' or i[4:9]=='03Li3' or i[4:9]=='04Ri3' or i[4:9]=='05Ri3' or i[4:9]=='06LR1' or i[4:9]=='07LR1' or i[4:9]=='08RR1' or i[4:9]=='09RR1' or i[4:9]=='10RR1':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(1)
  #         ponies.append(i[1:3])

  # # Grooved
  #     if i[4:9]=='02LR2' or i[4:9]=='03LR2' or i[4:9]=='04RR2' or i[4:9]=='05RR2' or i[4:9]=='06Li4' or i[4:9]=='07Li4' or i[4:9]=='08Ri4' or i[4:9]=='09Ri4' or i[4:9]=='10Ri4' or i[4:9]=='02Li4' or i[4:9]=='03Li4' or i[4:9]=='04Ri4' or i[4:9]=='05Ri4' or i[4:9]=='06LR2' or i[4:9]=='07LR2' or i[4:9]=='08RR2' or i[4:9]=='09RR2' or i[4:9]=='10RR2':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(0)
  #         ponies.append(i[1:3])
  # # Grooved Kissing
  #     if i[4:9]=='02LR1' or i[4:9]=='03LR1' or i[4:9]=='04RR1' or i[4:9]=='05RR1' or i[4:9]=='06Li3' or i[4:9]=='07Li3' or i[4:9]=='08Ri3' or i[4:9]=='09Ri3' or i[4:9]=='10Ri3' or i[4:9]=='02Li3' or i[4:9]=='03Li3' or i[4:9]=='04Ri3' or i[4:9]=='05Ri3' or i[4:9]=='06LR1' or i[4:9]=='07LR1' or i[4:9]=='08RR1' or i[4:9]=='09RR1' or i[4:9]=='10RR1':
  #         X.append(spectra[Names.index(i)].values.tolist())
  #         Y.append(1)
  #         ponies.append(i[1:3])


  # ======================================= k-fold ============================================
  accuracies=[]

  if method=='K-fold':
    for j in ['30', '33', '35', '37', '40', '41', '42', '47', '64']:
      X_train=[]
      X_test=[]
      Y_train=[]
      Y_test=[]
      for p in range(len(ponies)):
        if ponies[p]==j:
          X_test.append(X[p])
          Y_test.append(Y[p])
        else:
          X_train.append(X[p])
          Y_train.append(Y[p])
      # print('test:', Y_test)
      SVM_clf.fit(X_train,Y_train)
      y_pred_SVM = SVM_clf.predict(X_test)
      # print('pred:', y_pred_SVM)
      accuracies.append(accuracy_score(Y_test, y_pred_SVM))
    print(sum(accuracies)/9)



  # ======================================= Monte-Carlo ============================================
  accuracies=[]
  import random
  import statistics



  if method=='MC':
    print('How many pony groups for test? ')
    HowmanyPony=1
    for k in range(500):
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
        
        SVM_clf.fit(X_train,Y_train)
        y_pred_SVM = SVM_clf.predict(X_test)
        # print('pred:', y_pred_SVM)
        accuracies.append(accuracy_score(Y_test, y_pred_SVM))
        accuracies.sort()

    # print(sum(accuracies)/len(accuracies))
    # print(accuracies)
    print(statistics.median(accuracies))
  
