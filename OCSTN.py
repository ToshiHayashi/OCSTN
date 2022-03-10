#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import sys
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

from PIL import Image
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score,log_loss, mean_absolute_error,median_absolute_error
#from tqdm import tqdm
import pickle
def makefile(what,filename):
    with open(filename,"wb") as f3:
        pickle.dump(what,f3)

def readfile(filename):
    with open(filename,"rb") as f4:
        ans=pickle.load(f4)
    return ans


# In[ ]:


#from keras.datasets import cifar10
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.backend import clear_session
from tensorflow.keras import regularizers

table=[]
table2=[]
for time in range(5):
    train=readfile("X.pickle")
    train_label=readfile("y.pickle")
    train, x_test, train_label, y_test = train_test_split(train, train_label, test_size=0.2, random_state=time)
    #train=train.reshape(len(train)*2,330)
    #train_label=np.concatenate([train_label.reshape(-1,1),train_label.reshape(-1,1)],axis=1).reshape(len(train_label)*2)
    #x_test=x_test.reshape(len(x_test)*2,330)
    #y_test=np.concatenate([y_test.reshape(-1,1),y_test.reshape(-1,1)],axis=1).reshape(len(y_test)*2)

    M=train.shape[1]
    C=1

    I=np.zeros([M])
    
    #Goal Signal
    for i in range(M):
        #I[i]=i/M
        I[i]=i**2/M**2
        #I[i]=0.5
        #I[i]=1-(i-330)**2/330**2
        #I[i]=i%25/24
        #I[i]=((i-330)**3/330**3)/2+0.5
        





    act="linear"


    fil=16
    bat=16
    siz=25
    Q=5
    
    AUC_score=np.zeros(20)
    clear_session()
    single=np.zeros([20,5])
    for i in range(20):
        clear_session()
        Etest=np.zeros(len(x_test))
        acc=np.zeros(5)
        for n in range(5):
            clear_session()
            x_train=train[train_label==i]
            I_train=np.zeros([len(x_train),M])
            #I_train=np.zeros([len(x_train),M])
            for j in range(len(x_train)):

                I_train[j]=I
            I_train=I_train
            
            x_test=x_test.reshape(len(x_test),M,C)
            #x_test=x_test.reshape(len(x_test),M,C)
            y_label=np.zeros(len(y_test))
            y_label[y_test==i]=1
            # バックエンドに依存したチャネルの位置を調整する
            if K.image_data_format() == 'channels_last':
                x_train = x_train.reshape(x_train.shape[0],M,C)
                I_train = I_train.reshape(I_train.shape[0],M,C)
                
                input_shape = (M,C)
                #input_shape = (M,C)
            else:
                x_train = x_train.reshape(x_train.shape[0],
                                          C, M, N)
                x_test = x_test.reshape(x_test.shape[0],C, M, N)
                input_shape = (C, M, N)

            x_train = x_train.astype('float32')
            I_train = I_train.astype('float32')
            
            x_train, x_valid, I_train, I_valid = train_test_split(x_train, I_train, test_size=0.3, random_state=n)


            # model f:X → I
            model = models.Sequential()
            # Repeat convolution layer Q times
            model.add(layers.Conv1D(fil,kernel_size=siz,activation=act,padding='same',input_shape=input_shape))          
            for q in range(Q):
                model.add(layers.Conv1D(fil,kernel_size=siz,activation=act,padding='same'))           
            model.add(layers.Conv1D(C, kernel_size=siz,activation=act, padding='same'))
            model.compile(optimizer='Adam',
                          loss='mean_absolute_error')
            fit_callbacks = [
                callbacks.EarlyStopping(monitor='val_loss',
                                        patience=5,
                                        mode='min')
            ]
            #model.summary()
            # Train model
            model.fit(x_train, I_train,
                      epochs=100,
                      batch_size=bat,
                      shuffle=True,
                      validation_data=(x_valid, I_valid),callbacks=fit_callbacks,verbose=0)
            #modelname="model3/model"+str(i)+"_"+str(n)+".h5"
            #model.save(modelname)
            
            
            f_test=model.predict(x_test)
            I_train=I_train.reshape(I_train.shape[0],M ,C)
            #I_train=I_train.reshape(I_train.shape[0],M ,C)
            goal_image=I_train[0]
            #f_threshold=f_threshold.reshape(f_threshold.shape[0],int(M*N),C)
            f_test=f_test.reshape(f_test.shape[0],M,C)
            #f_test=f_test.reshape(f_test.shape[0],M,C)
            # equation (12)
            
            Etest2=np.zeros(len(Etest))
            for num in range(len(f_test)):
                Etest2[num]=mean_absolute_error(goal_image,f_test[num])
            Etest+=Etest2
            m=0
            single[i][n]=roc_auc_score(y_label,(-1)*Etest2)
            print(roc_auc_score(y_label,(-1)*Etest2))
            #print(roc_auc_score(y_label,(-1)*Etest))
        #multiply construction error by (-1)
        print(i)
        print(roc_auc_score(y_label,(-1)*Etest))
        AUC_score[i]=roc_auc_score(y_label,(-1)*Etest)
    #AUC.append(AUC_score)
    table2.append(AUC_score)
    for n in range(5):
        table.append(single[:,n])
    #table.append(AUC)
    #print(table)
    #print(table2)
a=np.array(table).mean(axis=0)
b=np.array(table).std(axis=0)
for i in range(20):
    print(round(a[i]*100,1),"±",round(b[i]*100,1))
print(round(a.mean()*100,1))

a=np.array(table2).mean(axis=0)
b=np.array(table2).std(axis=0)
for i in range(20):
    print(round(a[i]*100,1),"±",round(b[i]*100,1))
print(round(a.mean()*100,1))
#print(model_list)
#makefile(model_list,"model_list.pkl")


# In[ ]:




