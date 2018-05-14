import numpy as np
import matplotlib.pyplot as plt

def generateGaussianData(mean,covariance,amountData=100):
    return np.matrix([np.random.multivariate_normal(mean, covariance) for i in range (0,amountData)])

"""Plots a 2d scatter plot for matrix"""
def scatterPlot2DMatrix(data,pointsColor): 
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    for i in range (0,data.shape[0]):
        plt.scatter(data[i,0],data[i,1],color=pointsColor)
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    plt.show()