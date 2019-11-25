
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#Today we are going to implement the Naive Bayes classifier in python language and test it on
#the Pima Indians Diabetes dataset. While you are free to make your own implementation, it is
#recommended to follow the implementation steps below and test each one to make sure it
#works properly.
#1. Handle Data: Load the data from CSV file and split it into training and test datasets.
#The dataset is available at:
data=pd.read_csv("pima.csv")

def shuff(data,split_percentage):
    """The function shuffles the given data set and returns the data as DataFrame"""
    data=data.values
#    np.random.shuffle(data)
    train,test=np.split(data,[int(split_percentage*len(data))])
    x_train=train[:,0:-1]
    y_train=train[:,-1]
    x_test=test[:,0:-1]
    y_test=test[:,-1]
    return [x_train,y_train,x_test,y_test]


#2. Summarize Data (train): summarize the properties in the training dataset by
#calculate for every feature and class (prediction value) the mean and the std.
    
def statistics(x,y):
    train=pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1,sort=False)
    meu=train.groupby(train.iloc[:,-1]).mean()
    sigma=train.groupby(train.iloc[:,-1]).std()
    stats=np.array([(meu.iloc[0,:-1]),(meu.iloc[1,:-1]),(sigma.iloc[0,:-1]),(sigma.iloc[1,:-1])])
    return stats

#3. Write a function which make a prediction: Use the summaries of the dataset to
#generate a single prediction, which based on the gaussian distribution with the
#corresponding mean and std of each of the features. You can find the equation for
#the probability of an event given a Gaussian distribution in:
#https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
def gauss(x,m,sigma):
    Norm=np.sqrt(2*np.pi*(sigma**2))
    p=np.exp(-(((x-m)**2)/(2*(sigma**2))))/Norm
    return p

    
#4. Make Predictions: Generate predictions on the whole test dataset.
def probability(x,p_label,meu,sigma):
    probability=p_label
    for i in range(len(meu)):
        p=gauss(x[i],meu[i],sigma[i])
        probability*=p
    return probability


def predictions(x_train,y_train,x_test):
    predictions=np.zeros(len(x_test))   
    p1=np.sum(y_train)/len(y_train)
    p0=1-p1
    stats=statistics(x_train,y_train)
    for i in range(len(x_test)):
        P0=probability(x_test[i,:],p0,stats[0],stats[2])
        P1=probability(x_test[i,:],p1,stats[1],stats[3])
        if P1>P0:
            predictions[i]=1 
    return predictions


#5. Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as
#the percentage correct out of all predictions made.

def accuracy(predictions,y):
    return (100-(sum(abs(predictions-y))/len(y))*100)

#6. Tie it Together: Use all of the code elements to present a complete and standalone
#implementation of the Naive Bayes algorithm.

def NB_complete_implementation(data,split_percentage):
    data_set=shuff(data,split_percentage)
    x_train=data_set[0];y_train=data_set[1]
    x_test=data_set[2];y_test=data_set[3]
    test_predictions=predictions(x_train,y_train,x_test)
    final_answer=accuracy(test_predictions,y_test) 
    return f"The model is {final_answer:.2f} % accurate"

    
#* (Optional) Try building it into a class with fit(train) method which calculates the
#mean and std, and predict(test) method which makes a Naive Bayes prediction for
#the test data.
#class NB():
#    pass 
#    def __init__(self,data,split_percentage):
#        self.data=data
#        self.split_percentage=split_percentage
#    def data_split(self):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#We are going along the instructions from the following link:
#http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/