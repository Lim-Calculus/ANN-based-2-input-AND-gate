import numpy as np
import matplotlib.pyplot as plt

X_train=np.array([[0,0,1,1],[0,1,0,1]]) #size(2,4)
Y_train=np.array([[0,0,0,1]]) #(1,4)
def relu(x):
    y=np.maximum(0,x)
    return y

def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x,deriv=False):
        if (deriv==True):
            return x*(1-x) ##sigmoid activation functions
        return 1/(1+np.exp(-x))
def tanhh(x,deriv=False):
        if (deriv==True):
            return 1-np.power(np.tanh(x),2)##tanh activation functions
        return np.tanh(x)
def model(X,Y,learning_rate=0.05,iteration=20000):
    np.random.seed(1)
    W0=np.random.randn(2,X.shape[0])*np.sqrt(1/2) # size=(2,2)
    W1=np.random.randn(12,2)*np.sqrt(1/2) # size=(12,2)
    W2=np.random.randn(1,12)*np.sqrt(1/12) # size=(1,12)
    b0=np.zeros((2,1))
    b1=np.zeros((12,1))
    b2=np.zeros((1,1))
    m=X.shape[1]
    cost=[]
    for j in range(iteration):
        Z0=np.dot(W0,X)+b0 #(2,4)
        A0=relu(Z0) #size(2,4)
        Z1=np.dot(W1,A0)+b1#(12,4)
        A1=tanhh(Z1,deriv=False) #(12,4)
        Z2=np.dot(W2,A1)+b2 # size(1,4)
        A2=sigmoid(Z2,deriv=False)#size(1,4)       
        
        ## Calculate the error
        Cost=0.5*np.power(A2-Y,2)
        Error=A2-Y #Deriv Error
        dA2=Error #size(1,4)
        dZ2=dA2*(A2)*(1-A2) #size(1,4)
        dW2=(1/m)*np.dot(dZ2,np.transpose(A1))#size(1,12)
        db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True) #size(1,1)
        dA1=np.dot(np.transpose(W2),dZ2) #size (12,4)
        dZ1=dA1*(1-np.power(A1,2)) #size (12,4)
        dW1=(1/m)*np.dot(dZ1,np.transpose(A0)) #(12,4)*(4,2)==(12,2)
        db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True) 
        dA0=np.dot(np.transpose(W1),dZ1) #(2,12)*(12,4)==(2,4)
        dZ0=dA0*reluDerivative(Z0)#(2,4)
        dW0=(1/m)*np.dot(dZ0,np.transpose(X))#(2,4)*(4,2)==(2,2)
        db0=(1/m)*np.sum(dZ0,axis=1,keepdims=True) #(2,1)
        
        W2=W2-(learning_rate)*dW2
        b2=b2-(learning_rate)*db2
        W1=W1-(learning_rate)*dW1
        b1=b1-(learning_rate)*db1
        W0=W0-(learning_rate)*dW0
        b0=b0-(learning_rate)*db0      
        cost.append(np.mean(np.abs(Cost)))
        
        
        
        if j%1000==0:
            print(f"Estimated cost= {np.mean(np.abs(Cost))}")
            
    plt.plot(cost)
    plt.ylabel("cost")
    plt.xlabel("ëpochs (per 1000 iteration)")
    plt.title(f"Learning rate ={learning_rate}")
    plt.show()
    return W2,b2,W1,b1,W0,b0

def predict(X_test,Y_test,W2,b2,W1,b1,W0,b0):
    m = X_test.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    
    Z0=np.dot(W0,X_test)+b0 #(2,4)
    A0=relu(Z0) #size(2,4)
    Z1=np.dot(W1,A0)+b1#(12,4)
    A1=tanhh(Z1,deriv=False) #(12,4)
    Z2=np.dot(W2,A1)+b2 # size(1,4)
    A2=sigmoid(Z2,deriv=False)#size(1,4)
  
    print(A2)
    for i in range(0, A2.shape[1]):
        if A2[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == Y_test[0,:]))))
    

    # print results
   
    
    return p

## Testing
X_test=np.array([[0,1],[0,1]]) #size(2,1)``
Y_test=np.array([[0,1]]) #size(1,1)
W2,b2,W1,b1,W0,b0=model(X_train,Y_train,learning_rate=0.05,iteration=200000)
predict(X_train,Y_train,W2,b2,W1,b1,W0,b0)

