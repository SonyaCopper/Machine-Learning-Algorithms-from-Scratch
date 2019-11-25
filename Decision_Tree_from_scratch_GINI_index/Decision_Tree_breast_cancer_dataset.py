
import numpy as np
import pandas as pd

data_main=pd.read_csv("wdbc.csv",header=0)
"ID number 2) Diagnosis (M = malignant, B = benign) a) radius b) texture c) perimeter d) area e) smoothness f) compactness g) concavity h) concave points i) symmetry j) fractal dimension"

data_main.iloc[:,1]=(data_main.iloc[:,1].replace(["M","B"],[1,0]))
data_main=data_main.drop(data_main.columns[0],axis=1)
data_main=data_main.values
data_main[:,(0,-1)]=data_main[:,(-1,0)] 
#np.random.shuffle(data_main)
split_percentage=0.8
train,test=np.split(data_main,[int(split_percentage*len(data_main))])


def get_split(data):
    
    gini_matrix=np.zeros((data.shape[0],data.shape[1]-1))
    
    for i in range(data.shape[1]-1):
        
        for j in range(data.shape[0]):
            
            value=data[j,i]
            b=split(data,i,value)
            gini_matrix[j,i]=gini(b[0])*(len(b[0])/len(data))+gini(b[1])*(len(b[1])/len(data))
            gin=np.unravel_index(np.argmin(gini_matrix, axis=None), gini_matrix.shape)
            branches=split(data,gin[1],data[gin[0],gin[1]])
            
    return branches[0],branches[1],gin[1],data[gin[0],gin[1]]


def gini(branch):
    
    if len(branch)==0:
        gini=0
        
    else:
        ones=np.count_nonzero(branch[:,-1])/len(branch)
        gini=1-(ones**2+(1-ones)**2)
        
    return gini



def split(data,i,value):
    
    right=data[data[:,i]<=value];
    left=data[data[:,i]>value]
    
    return [left,right]


 
class Node():
    
    
    def __init__(self,data,depth):
    
        self.depth=depth
        self.data=data
        self.right=None
        self.left=None  
        self.feature_index=None
        self.feature_value=None
        self.label=None
        self.max_depth=5
        self.min_length=5
        
        
    def split_node(self):
        
        if self.depth >self.max_depth or len(self.data)<self.min_length or gini(self.data)==0:
            self.label=self.leaf()
            return 
        
        else:
            self.left,self.right,self.feature_index,self.feature_value=get_split(self.data)
            
            self.left=Node(self.left,self.depth+1)
            self.left.split_node()
            
            self.right=Node(self.right,self.depth+1)
            self.right.split_node()
            
            

    def leaf(self):
        
        ones=np.count_nonzero(self.data[:,-1])
        zeros=len(self.data)-ones
        
        if zeros<=ones:  
            return 1
        
        else:   
            return 0
       
          
    def evaluation_point(self,test):
        
        if self.label !=None:
            
            return self.label
        
        else:
            
            if test[self.feature_index] < self.feature_value:
                
                return self.right.evaluation_point(test)
            
            else: 
                
                return self.left.evaluation_point(test)
            
              
def decision_tree(train,depth,test): 
    
    root=Node(train,depth)
    root.split_node()   
    predictions=np.zeros(len(test))
    
    for i in range(len(test)):
        predictions[i]=root.evaluation_point(test[i])
    
    return predictions
    
def accuracy(test,predictions):
    
    accuracy=100-(sum(abs(test[:,-1]-predictions))/len(predictions))*100
    
    return f"{accuracy:.2f} % of the test data was predicted correctly"











        