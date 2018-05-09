import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self,neuralLayers):
        self.numLayers=len(neuralLayers)-1
        self.w0 = []
        self.W = []
        self.lossArray=[]
        for i in range(self.numLayers):
            self.W.append(np.random.randn(neuralLayers[i], neuralLayers[i+1]) / np.sqrt(neuralLayers[i]))
            self.w0.append(np.random.randn(neuralLayers[i+1]).reshape(1, neuralLayers[i+1]))
                
    def calculateLoss(self,X, y):
        softmaxLayer = SoftmaxLayer()
        x_input = X
        for i in range(len(self.W)):        
            x_input=self.forwardPropogation(x_input,i)
        return softmaxLayer.calculateLoss(x_input, y)
    
    def forwardPropogation(self,z,currentLayer):
        a=(-1)*(np.dot(z, self.W[currentLayer])+self.w0[currentLayer])
        return sigmoid(a)
    
    def backPropogation(self,forward,topError,currentLayer,regularizationParam,epsilon): 
        
        topError=(1.0 - forward[currentLayer]) * forward[currentLayer] * topError
        matrixOfOnes=topError * np.ones_like(forward[currentLayer]) 
        dW = forward[currentLayer-1].T.dot (matrixOfOnes)
        dW += regularizationParam * self.W[currentLayer-1]
        dw0 = np.ones((1, topError.shape[0]), dtype=np.float64).dot( topError)
        propagationError = matrixOfOnes.dot(self.W[currentLayer-1].T)
        self.w0[currentLayer-1] += -epsilon * dw0
        self.W[currentLayer-1] += -epsilon * dW
        return dW,dw0,propagationError
    
    def train(self,x,t,numberIterations=1000,epsilon=0.01,regularizationParam=1):
        for epoch in range(numberIterations):
            # Forward propagation
            x_input = x
            X = [x_input]
            for i in range(self.numLayers):
                x_input=self.forwardPropogation(x_input,i)
                X.append(x_input)    
            # Back propagation
            softmaxLayer = SoftmaxLayer()
            propogationError = softmaxLayer.findDerivative(X[-1], t)
            for i in range(self.numLayers, 0, -1): 
                dW,dw0,propogationError=self.backPropogation(X,propogationError,i,regularizationParam,epsilon)
            loss=self.calculateLoss(x, t)
            if epoch%100==0:
                self.lossArray.append(loss)
                
    def predict(self, X):
        softmaxLayer = SoftmaxLayer()
        x_input = X
        for i in range(len(self.W)):
            x_input=self.forwardPropogation(x_input,i)
        probs = softmaxLayer.predict(x_input)
        return np.argmax(probs, axis=1)

    def plotPredictionCurve(self,X,t):
        function=lambda x: self.predict(x)
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z,alpha=.7, cmap=plt.cm.RdBu)
        plt.title("Decision Boundary for Given Function")
        plt.show()
        
    def plotCostHistory(self):
        x=[i for i in xrange (len(self.lossArray))]
        plt.figure()
        plt.plot(x,self.lossArray)
        plt.title("Cost History of Training")
        
class SoftmaxLayer:
    def predict(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def calculateLoss(self, X, y):
        probs = self.predict(X)
        corect_logprobs = -np.log(probs[range(X.shape[0]), y])
        return 1./X.shape[0] * np.sum(corect_logprobs)

    def findDerivative(self, X, y):
        probs = self.predict(X)
        probs[range(X.shape[0]), y] -= 1
        return probs

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(x))