"""
amounnt of battery, brightness of screen = Device use time left
amount of snow from 0-10 inches
temperature 1-50 degrees
chance of school closure = (10*(10*S))- ((2*T)*3)
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn

def generateData():
    N = 1000
    data = np.random.random((N,3))
    for i in range(len(data)):
        data[i][0] = math.ceil(10 * data[i][0])
        data[i][1] = math.ceil(50 * data[i][1])
        if np.random.random() <= 0.15:
            #outlier
            data[i][2] = ((10 * math.ceil(10 * np.random.random())) * 10) - ((2 * math.ceil(50 * np.random.random())) * 3)
        else:
            data[i][2] = ((10 * data[i][0]) * 10) - ((2 * data[i][1]) * 3)

    np.savetxt("Data/best4linreg.csv", data, delimiter=",")


def test():
    inf = open('Data/best4linreg.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    # create a learner and train it
    learner = lrl.LinRegLearner() # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in of sample
    predYLRL = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predYLRL) ** 2).sum()/trainY.shape[0])
    print
    print "Linear Regression Learner"
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predYLRL, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predYLRL = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predYLRL) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predYLRL, y=testY)
    print "corr: ", c[0,1]

    learner = knn.KNNLearner(k = 3) # constructor
    learner.addEvidence(trainX, trainY) # training step

    predYKNN = learner.query(trainX) # query
    rmse = math.sqrt(((trainY - predYKNN) ** 2).sum()/trainY.shape[0])
    print
    print "KNN Learner"
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predYKNN, y=trainY)
    print "corr: ", c[0,1]

    predYKNN = learner.query(testX) # query
    rmse = math.sqrt(((testY - predYKNN) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predYKNN, y=testY)
    print "corr: ", c[0,1]



if __name__=="__main__":
    generateData()
    test()
