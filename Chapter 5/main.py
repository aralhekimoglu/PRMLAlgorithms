import matplotlib.pyplot as plt
import sklearn.datasets
from NeuralNetwork import NeuralNetwork

x,t= sklearn.datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
plt.figure()
plt.scatter(x[:,0], x[:,1], s=40, c=t, cmap=plt.cm.RdBu)  

neuralLayers = [2,3,2]
neuralNetwork=NeuralNetwork(neuralLayers)
neuralNetwork.train(x,t,numberIterations=20000,regularizationParam=0.01)
neuralNetwork.plotPredictionCurve(x,t)
neuralNetwork.plotCostHistory()