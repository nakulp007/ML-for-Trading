"""
import BagLearner as bl
learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 20, boost = False)
learner.addEvidence(Xtrain, Ytrain)
Y = learner.query(Xtest)
"""

import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False):
        self.learner = learner
        self.numBags = bags
        self.boost = boost
        self.learners = []
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self,dataX,dataY):
        def getBags(dataX, dataY):
            bgs = []
            for b in range(self.numBags):
                indexesToSample = np.random.choice(list(xrange(len(dataY))), dataY.size, True)
                sampleX = dataX[indexesToSample]
                sampleY = dataY[indexesToSample]
                bgs.append((sampleX,sampleY))
            return bgs

        bags = getBags(dataX, dataY)
        for i in range(0, len(self.learners)):
            self.learners[i].addEvidence(bags[i][0], bags[i][1])

    def query(self,points):
        allLearnerAnswers = []
        for i in range(0, len(self.learners)):
            allLearnerAnswers.append(self.learners[i].query(points))
        allLearnerAnswers = np.array(allLearnerAnswers)
        meanAllLearnerAnswers = np.mean(allLearnerAnswers, axis=0)
        return meanAllLearnerAnswers

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
