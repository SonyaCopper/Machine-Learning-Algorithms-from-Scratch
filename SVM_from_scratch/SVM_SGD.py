
import numpy as np
import pandas as pd 
data=pd.read_csv("pima.csv")
data.iloc[:,-1]=(data.iloc[:,-1].replace(0,-1))
data=data.values
np.random.shuffle(data)
train,test=np.split(data,[int(0.8*len(data))])
x_train=train[:,0:-1]
y_train=train[:,-1]
x_test=test[:,0:-1]
y_test=test[:,-1]
def norm(x):
    for i in range(x.shape[1]):
        meu=np.mean(x,axis=0)
        sigma=np.std(x,axis=0)
        x[:,i]=(x[:,i]-meu[i])/sigma[i]
    return x

x_train=norm(x_train)
x_test=norm(x_test)

def add_bias(x):
    bias=np.ones(len(x))
    x=np.column_stack((bias,x))
    return x 
    
x_train=add_bias(x_train)
x_test=add_bias(x_test)

class SVM_SGD():
    def __init__(self,x_train,y_train):
        self.x=x_train
        self.y=y_train
        self.iterations=100
        self.alpha=0.01
        self.w=np.zeros(len(self.x[0]),dtype=np.float64)
        self.w_vec=np.zeros((self.iterations,len(self.x[0])))
        self.tot_loss=np.zeros(self.iterations)

    def hinge_loss(self,score,y):
                return max(0,1-score*y)
            
    def gradient_discent(self):
        error=np.zeros(len(self.x))
        for i in range(self.iterations):
            for j in range(len(self.x)) :
                score=np.dot(self.x[j],self.w)
                if (self.y[j]*(score))<=1:
                    self.w+=self.alpha*((self.y[j]*self.x[j])) 
                else:
                    self.w+=0 
                error[j]=self.hinge_loss(score,self.y[j])
            self.tot_loss[i]=np.mean(error)
            self.w_vec[i]=self.w
        w=self.w_vec[np.argmin(self.tot_loss, axis=None)]
        return w
    
    def model_accuracy(self):
        score=np.sign(np.dot(self.x,self.gradient_discent()))
        accuracy=(np.sum((score==self.y))/len(self.y))*100
        return f"The accuracy of the model is {accuracy:.2f} %"
    
    def prediction(self,test):
        self.test=test
        self.prediction=np.sign(np.dot(self.test,self.gradient_discent()))
        return self.prediction

        

d=SVM_SGD(x_train,y_train)






















