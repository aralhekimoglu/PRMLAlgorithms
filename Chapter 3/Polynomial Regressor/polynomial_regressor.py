import random
import numpy as np
import math
import matplotlib.pyplot as plt

# generate poly data, take degree, amount of data to be generated and vertical 
# range as input
def generate_polynomial_data(degree=1, amount=10, x_range=10):
    y=[]
    rng=np.arange(0.0, x_range, x_range/float(amount))
    x=np.matrix([i for i in rng]).transpose()
    for i in rng:
        powered_i=math.pow(i,degree)
        y.append(powered_i+random.uniform(-powered_i*0.1,powered_i*0.1))
    y=np.matrix(y).transpose()
    return x,y

#input an matrix and feature scale its columns (x-mean)/variance
def feature_scaling(X):
    X=X.astype(float)
    mean_X=[]
    var_X=[]
    col_size=X.shape[1]
    for i in range(0,col_size):
        mean_i=np.mean(X[:,i])
        mean_X.append(mean_i)
        var_i=np.var(X[:,i])
        var_X.append(var_i)
        if (var_i!=0):
            X[:,i]=(X[:,i]-mean_i)/float(var_i)
    return X,mean_X,var_X

# turn regular feature into its polynomial features
def polynomial_features(x,degree=2):
    row_size=x.size
    x_poly=np.zeros((row_size,degree+1))
    for i in range (0,row_size):
        for j in range (0, degree+1):
            x_poly[i,j]=math.pow(x[i,:],j)
    return x_poly

# plot the prediction of poly regressor
def plot_prediction(w,x_range=10):
    x_pred=np.matrix([i for i in range (0,x_range)]).transpose()
    print x_pred
    x_pred_poly=np.matrix(polynomial_features(x_pred,degree=w.size-1)) 
    print x_pred_poly
    for i in range (1,w.size):
        x_pred_poly[:,i]=(x_pred_poly[:,i]-mean_poly[i])/variance_poly[i]  
    y_pred=x_pred_poly.dot(w)    
    plt.plot(x_pred,y_pred,color='red',label='yes')

# these two are parameters that can be played with
poly_degree=3
range_x=100
lamb=0.005

# split generated data into train and test sets.
x,y=generate_polynomial_data(degree=poly_degree,amount=100,x_range=range_x)

# plot generated data
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')

# compute polynomial features for x
x_poly=polynomial_features(x,poly_degree)
x_poly_scaled,mean_poly,variance_poly=feature_scaling(x_poly)

# fit w to data
x_poly_scaled_t=x_poly_scaled.transpose()
xt_x=x_poly_scaled_t.dot(x_poly_scaled)
w=np.linalg.pinv(lamb*np.identity(xt_x.shape[1])+xt_x).dot(x_poly_scaled_t).dot(y)

# plot prediction line
plot_prediction(w,range_x)