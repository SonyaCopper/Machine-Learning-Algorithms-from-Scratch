

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

iris=pd.read_csv("iris1.csv")
iris.columns=["sepal_length","sepal_width","petal_length","petal_width","flower"]
iris=iris.drop(["flower"],axis=1)
data=iris.values
#np.random.shuffle(data)


def norm(x):
    """ Normalization of the data, assumed data contains only numerical values"""
    for i in range(x.shape[1]):
        meu=np.mean(x,axis=0)
        sigma=np.std(x,axis=0)
        x[:,i]=(x[:,i]-meu[i])/sigma[i]
    return x



def k_points(data,k,meu=0,sigma=1):
    """Returns matrix of k points, meu and sigma defined as 0 and 1 because the data is normalized"""
    points=np.zeros((k,data.shape[1]))
    for i in range(k):
        for j in range(data.shape[1]):
            points[i,j]=np.random.uniform((meu-2*sigma),(meu+2*sigma))
    return points


def dist(centers,point):
    """Calculate the distance beetween centers and point"""
    dist=np.zeros(centers.shape[0])
    for i in range(centers.shape[0]):
        dist[i]=np.sqrt(np.sum(np.power((centers[i]-point),2))) 
    return dist

def clusters(data,centers):
    clusters={i:[] for i in range(centers.shape[0])}
    for i in range(data.shape[0]):
        clusters[np.argmin(dist(centers,data[i]))].append(data[i])
    return clusters

def means(clusters,old_centroids):
    new_centroids=[]
    for i in range(len(clusters)):
        if len(clusters[i])==0:
            new_centroids.append(old_centroids[i])
        else:
            new_centroids.append(np.mean(clusters[i],axis=0))
    return np.array(new_centroids)

def check_exit(old_centroids,new_centroids):
    return np.sqrt(np.sum(np.power((old_centroids-new_centroids),2)))

def graph(clusters):
    c=["r","g","b"]
    for i in range(len(clusters)):
        if len(clusters[i])==0:
            continue
        else:
            x=(np.array(clusters[i])).T[0]
            y=(np.array(clusters[i])).T[1]
            plt.scatter(x,y,color=c[i])
    return

def k_means(data):
    data=norm(data)
    old_centroids=k_points(data,3)
    new_centroids=k_points(data,3)
    while check_exit(old_centroids,new_centroids)!=0:
        old_centroids=new_centroids
        new_clusters=clusters(data,new_centroids)
        new_centroids=means(new_clusters,new_centroids)
        plt.show(graph(new_clusters))
    return new_clusters

tt=k_means(data)





















          