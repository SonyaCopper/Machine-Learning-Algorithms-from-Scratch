# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data

K=3
meu_0=data[np.random.randint(data.shape[0],size=K)]
sigma_0=np.array([np.cov(data.T)]*K)
alpha_0=np.ones(K)*(1/K)

def gaussian(x,meu_k,sigma_k):
    m=x-meu_k
    f=np.linalg.inv(sigma_k)
    norm=1/(np.sqrt(((2*np.pi))*(np.linalg.det(sigma_k))))
    exp=np.exp(-0.5*(m.T.dot(f.dot(m))))
    return norm*exp

def get_new_sigma(data,new_meu,w,N_k):
    sigma_tot=np.array([np.zeros((data.shape[1],data.shape[1]))]*K)
    for k in range(K):
        sum_sigma_i=0
        for i in range(len(data)):
            x=data[i]-new_meu[k]
            sigma=w[i,k]*np.outer(x,x.T)
            sum_sigma_i+=sigma
        sigma_tot[k]=sum_sigma_i/N_k[k]
    return sigma_tot

def EM(data,alpha,meu,sigma,iterations):
    w=np.zeros((data.shape[0],K))
    total_likelihood=np.zeros(iterations)
    count=0
    while count<iterations:
        likelihood=0
        for i in range(data.shape[0]):
            for k in range(K):
                w[i,k]=(alpha[k]*gaussian(data[i],meu[k],sigma[k]))          
            total_weights_per_row=np.sum(w[i])
            w[i]=w[i]/total_weights_per_row
            likelihood+=np.log(total_weights_per_row)
        total_likelihood[count]=likelihood
        N_k=np.sum(w,axis=0)
        alpha=N_k/(data.shape[0])
        meu=(data.T.dot(w)/N_k).T
        sigma=get_new_sigma(data,meu,w,N_k)
        count+=1
    return N_k,meu,sigma,total_likelihood,w
def creat_clusters(data,weight):
    labels=np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        labels[i]=np.argmax(weights[i])
    label=range(K)
    clusters=[0]*K
    x=np.column_stack((data,labels))
    for i in range(len(label)):
        clusters[i]=x[x[:,-1]==label[i]]
    return clusters


def graph(clusters):
    cl=["r","g","b","k","y"]
    for i in range(len(clusters)):
        plt.scatter(clusters[i][:,2],clusters[i][:,3], color=cl[i])       
    
n_of_points_in_cluster,meu,sigma,likelihood,weights=EM(data,alpha_0,meu_0,sigma_0,20)
x=np.arange(len(likelihood))
plt.figure(1)
plt.scatter(x,likelihood)
plt.title("log(likelihood) vs iterations")
c=creat_clusters(data,weights)
plt.figure(2)
graph(c)
plt.title("clusters")


    
    
    
    
    

