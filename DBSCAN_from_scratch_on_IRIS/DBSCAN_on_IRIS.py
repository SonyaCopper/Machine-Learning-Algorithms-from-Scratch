

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

iris=pd.read_csv("iris1.csv")
iris.columns=["sepal_length","sepal_width","petal_length","petal_width","flower"]

iris=iris.drop(["flower"],axis=1)
data=iris.values

def norm(x):
    """ Normalization of the data, assumed data contains only numerical values"""
    for i in range(x.shape[1]):
        meu=np.mean(x,axis=0)
        sigma=np.std(x,axis=0)
        x[:,i]=(x[:,i]-meu[i])/sigma[i]
    return x
data=norm(data)
epsilon=0.76
Pmin=data.shape[1]+1

def dbscan(data,epsilon,Pmin):
    labels=np.zeros(len(data))
    cluster_name=1
    for i in range(data.shape[0]):
        if labels[i]==0 or labels[i]==-1:
            neighbours,ix=dist(data,data[i],labels,cluster_name)
            if len(neighbours)<Pmin:
                labels[i]=-1
            else: 
                labels[ix]=cluster_name
                labels=expand(neighbours,data,labels,cluster_name,Pmin)         
                cluster_name+=1
    return labels

def dist(data,point,labels,cluster_name):
    """Calculate the distance beetween point and neighbours in radius epsilon, return the neighbours and insexes of the points"""
    neighbours=[]
    indexes=[]
    for i in range(data.shape[0]):
        if labels[i]==0 or labels[i]==-1:
            dist=np.sqrt(np.sum(np.power((data[i]-point),2))) 
            if dist<=epsilon:
                neighbours.append(data[i])
                indexes.append(i)
    return neighbours,indexes

def expand(neighbours,data,labels,cluster_name,Pmin):

        for point in neighbours:
            new_n,ix=dist(data,point,labels,cluster_name)
            labels[ix]=cluster_name
            if len(new_n)>=Pmin:
                labels=expand(new_n,data,labels,cluster_name,Pmin) 
        return labels

def clusters(data,labels):
    label=list(set(labels))
    clusters=[0]*len(label)
    x=np.column_stack((data,labels))
    for i in range(len(label)):
        clusters[i]=x[x[:,-1]==label[i]]
    return clusters

def graph(clusters):
    cl=["r","g","b","k","y"]
    for i in range(len(clusters)):
        plt.scatter(clusters[i][:,2],clusters[i][:,3], color=cl[i])
        
        
db=dbscan(data,epsilon,Pmin)   
c=clusters(data,db)
g=graph(c)

labels_sl = DBSCAN(eps=0.44, min_samples=5).fit_predict(data)
clusters_sl=clusters(data,labels_sl)
g_sl=graph(clusters_sl)

### the two results are similar, however skilearn algorithem uses smaler epsilon for same clusttering 
###fit_predict(X) calls fit(X) fit_predict(X) returns cluster labels while fit(X) only preforms the clusttering.
### fit_predict(X) gives the same as calling fit(X) first and calling .labels_ 