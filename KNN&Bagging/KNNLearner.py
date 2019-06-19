"""
import KNNLearner as knn
learner = knn.KNNLearner(k = 3) # constructor
learner.addEvidence(Xtrain, Ytrain) # training step
Y = learner.query(Xtest) # query
"""

import numpy as np

class KNNLearner(object):

    def __init__(self, k=3):
        self.k = k

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        """
        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        newdataX[:,0:dataX.shape[1]]=dataX

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        """
        self.dataX = dataX
        self.dataY = dataY


    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        trainDataSetSize = self.dataY.size
        returnY = []
        #go throught each test case
        for point in points:
            differenceMatrix = np.tile(point, (trainDataSetSize,1)) - self.dataX
            sqDifferenceMatrix = differenceMatrix**2
            sqDistances = sqDifferenceMatrix.sum(axis=1)
            distances = sqDistances**0.5
            sortedDistIndicies = distances.argsort()
            closestPoints = []
            for i in range(self.k):
                closestPoints.append(self.dataY[sortedDistIndicies[i]])
            returnY.append(np.mean(closestPoints))
        return np.array(returnY)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
