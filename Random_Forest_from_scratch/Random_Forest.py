

import numpy as np
import pandas as pd
from Decision_tree_for_RF import decision_tree,Node
data_main=pd.read_csv("wdbc.csv",header=0)
data_main.iloc[:,1]=(data_main.iloc[:,1].replace(["M","B"],[1,0]))
data_main=data_main.drop(data_main.columns[0],axis=1)
data_main=data_main.values;data_main[:,(0,-1)]=data_main[:,(-1,0)] 
np.random.shuffle(data_main)
split_percentage=0.8
train,test=np.split(data_main,[int(split_percentage*len(data_main))])


def creat_subsample(data,n):
    subsample_index=np.random.randint(len(data),size=n)
    return data[subsample_index]

def random_forest(train,depth,N):
    trees=[]
    for i in range(N):
        baby_tree=creat_subsample(train,int(0.6*len(train)))
        root_tree=Node(baby_tree,1)
        trees.append(root_tree)
    return trees

def bagging_predict_point(trees,test):
    predictions_vec=[]
    for i in range(len(trees)):
        prediction=decision_tree(trees[i],test)
        predictions_vec.append(prediction)
    return max(predictions_vec,key=predictions_vec.count)

def bagging_predict_test_set(train,depth,test,N):
    g=[]
    for i in range(len(test)):
        y=bagging_predict_point(random_forest(train,depth,N),test[i])
        g.append(y)
    return g
#        
c=bagging_predict_test_set(train,1,test,5)
#g=test[:20]
#y=g[:,-1]
##b=bagging_predict(random_forest(train,1,5),test[6])
#def K_Fold(data,K):
#    K_parts=[]
#    for i in range(K):
#        d=data_main[i::K]
#        K_parts.append(d)
#    return K_parts
#    
#f=K_Fold(data_main,4)  
#def K_Fold_forest(data,K):
#    devided_data=K_Fold(data,K)
#    for i in range(K):
#        test=devided_data[i]
#        devided_data.remove(devided_data[i])
#        train=devided_data
#        
        
        
        
        
        
        


    










#def random_features(data):
#    random_index=np.random.choice(train.shape[1]-1,int(np.ceil(np.sqrt(train.shape[1]))),replace=False)
#    data=np.column_stack((data[:,random_index],data[:,-1]))
#    return data

        