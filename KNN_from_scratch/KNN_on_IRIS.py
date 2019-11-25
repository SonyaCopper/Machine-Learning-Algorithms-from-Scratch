# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:10:51 2019

@author: Sonya
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#iris=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
#pd.DataFrame(iris).to_csv("iris1.csv",index=False)
iris=pd.read_csv("iris1.csv")
iris.columns=["sepal_length","sepal_width","petal_length","petal_width","flower"]

def shuff(data,percentage_for_train):
    """The function shuffles the given data set and returns the data as DataFrame"""
    data=data.values
    np.random.shuffle(data)
    train,test=np.split(data,[int(percentage_for_train*len(data))])
    x=train[:,0:-1]
    y=train[:,-1]
    x_test=test[:,0:-1]
    y_test=test[:,-1]
    return [data,x,y,x_test,y_test]

data_set=shuff(iris,0.8)
x=data_set[1];y=data_set[2]
x_test=data_set[3][2];y_test=data_set[4][2] ###Taking one point for now 

def dis(x,x_test,y,y_test):
    f=0
    p={}
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
         s=(x[i][j]-x_test[j])**2
         f+=s
        p[i]=(f**0.5,y[i])
    return p
g=pd.DataFrame(dis(x,x_test,y,y_test)).T
g.columns=["distance","flower"]
K=9
c=g.head(K)
t=c.groupby("flower",as_index=False).count()
p=c.groupby("flower",as_index=False).sum()
f=t.flower.loc[t.distance==t.distance.max()]==y_test   
m=pd.DataFrame((t.flower,t.distance/p.distance)).T
a=pd.DataFrame((t.flower,t.distance/K)).T
f=m.flower.loc[m.distance==m.distance.max()]==y_test
aa=a.flower.loc[a.distance==a.distance.max()]==y_test
l=t.sort_values(by="distance").head(len(f))
#fb=f[0]
#if len(f)>1
#else:
#    continue

#def total(x_test,y_test):
#    final=np.zeros(len(x_test))
#    for i in range(x_test.shape[0]):
#        dis(x,y,x_test,y_test,K)
#        
    
    
    
    
    
    
    
    



    

#g["]
#g.sort()
#K=10
#k=g[:K]



