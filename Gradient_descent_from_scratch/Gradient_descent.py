
import numpy as np
import matplotlib.pyplot as plt

## Linear regration ####################################################################################################
old=np.array([31,22,40,26])
young=np.array([22,21,37,25])
times=np.array([2,3,8,12])
x0=np.column_stack((old,young))
diff=(old-young)
x_1=np.column_stack((x0,diff.T))
diff_p2=diff**2
x_2=np.column_stack((x0,diff_p2.T))
bias=np.ones(len(old))
x_3=np.column_stack((x0,bias))
x_4=np.column_stack((bias,x_2))   

def tetha(x,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x)),times)


h=[tetha(x0,times),tetha(x_1,times),tetha(x_2,times),tetha(x_3,times),tetha(x_4,times)]


def cost_function_normal(h,X,Y=times):
    m=len(h)
    predictions=X.dot(h)
    cost=(0.5/m)*np.sum(np.square(predictions-Y))
    return cost


lost={1:cost_function_normal(h[0],x0),2:cost_function_normal(h[1],x_1),3:cost_function_normal(h[2],x_2),4:cost_function_normal(h[3],x_3),5:cost_function_normal(h[4],x_4)}

##In the given case the hypothesis C x3 = (x1 − x2)^2 is given the best resolut Cost=3.51,The best result can be achived by also adding bias Cost:3.028e-21 


##Gradien discent#####################################################################################################

x=np.array([0,1,2])
Y=np.array([1,3,7])
m=len(x)
alpha=np.array([1, 0.1, 0.01])
vector_tetha=np.array([2,2,0],dtype=np.float64)
bias=np.ones(m)
x_2=x**2
X=np.column_stack((bias,x,x_2))

def cost_function(vector_tetha,X,Y):
    predictions=X.dot(vector_tetha)
    cost=(0.5/m)*np.sum(np.square(predictions-Y))
    return cost


def gradient_discent(vector_tetha,X,Y,alpha,iterations=200):
    vector_tetha=np.array([2,2,0],dtype=np.float64)
    tetha_history=np.zeros((iterations,len(vector_tetha)))
    cost_hystory=np.zeros(iterations)

    for i in range(iterations):
        predictions=np.dot(X,vector_tetha)
        vector_tetha-=(1/m)*alpha*(X.T.dot((predictions-Y)))
        tetha_history[i,:]=vector_tetha
        cost_hystory[i]=cost_function(vector_tetha,X,Y)
    results=[vector_tetha,tetha_history,cost_hystory]
    return results 


x_axis=np.arange(200)
plt.figure(1)                                      
plt.scatter(x_axis,gradient_discent(vector_tetha,X,Y,0.1)[2],label="LR=0.1")
plt.scatter(x_axis,gradient_discent(vector_tetha,X,Y,0.01)[2],label="LR=0.01")
plt.scatter(x_axis,gradient_discent(vector_tetha,X,Y,0.001)[2],label="LR=0.001")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title("Cost vs Learning rate")
plt.legend()
 ## For L.R=1, the cost function doesnt converge, the stap is too big for the given data
 ## as can be seen in Fig(1) the converging rate of the cost function is slower when the L.R is smaller 

#### Momentum###########################################################################################


def gradient_discent_momentum(vector_tetha,X,Y,alpha=0.1,iterations=200):
    vector_tetha=np.array([2,2,0],dtype=np.float64)
    tetha_history=np.zeros((iterations,len(vector_tetha)))
    cost_hystory=np.zeros(iterations)
    v_t_1=np.zeros(len(vector_tetha))
    gamma=0.9
    
    for i in range(iterations):
        predictions=np.dot(X,vector_tetha)
        v_t=gamma*v_t_1+(1/m)*alpha*(X.T.dot((predictions-Y)))
        vector_tetha-=v_t
        v_t_1=v_t
        tetha_history[i,:]=vector_tetha
        cost_hystory[i]=cost_function(vector_tetha,X,Y)
    
    results=[vector_tetha , tetha_history, cost_hystory]
    return results 

##Nesterov#####################################################################################################

def gradient_discent_Nesterov(vector_tetha,X,Y,alpha=0.1,iterations=200):
    vector_tetha=np.array([2,2,0],dtype=np.float64)
    tetha_history=np.zeros((iterations,len(vector_tetha)))
    cost_hystory=np.zeros(iterations)
    v_t_1=np.zeros(len(vector_tetha))
    gamma=0.9
    
    for i in range(iterations):
        predictions=np.dot(X,(vector_tetha-gamma*v_t_1))
        v_t=gamma*v_t_1+(1/m)*alpha*(X.T.dot((predictions-Y)))
        vector_tetha-=v_t
        v_t_1=v_t
        tetha_history[i,:]=vector_tetha
        cost_hystory[i]=cost_function(vector_tetha,X,Y)
        
    results=[vector_tetha , tetha_history, cost_hystory]
    return results

plt.figure(2)                                      
plt.scatter(x_axis,gradient_discent(vector_tetha,X,Y,0.1)[2],label="G.D")
plt.scatter(x_axis,gradient_discent_momentum(vector_tetha,X,Y,0.1)[2],label="Momentum")
plt.scatter(x_axis,gradient_discent_Nesterov(vector_tetha,X,Y,0.1)[2],label="Nesterov")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title("Cost vs Algorithem")
plt.legend()
## As we can see in Fig(2) the converging rate of the cost function is faster for Mometum and Nesterov as compered to classic Gradient discent   

    
    
    