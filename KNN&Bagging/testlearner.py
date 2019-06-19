"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl

if __name__=="__main__":
    inf = open('Data/ripple.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    #start my test
    import KNNLearner as knn
    knnktest = []
    for r in range(1,101):
        learner = knn.KNNLearner(k = r) # constructor
        learner.addEvidence(trainX, trainY) # training step
        predYKNNTrain = learner.query(trainX) # query
        rmseTrain = math.sqrt(((trainY - predYKNNTrain) ** 2).sum()/trainY.shape[0])
        print
        print "KNN Learner"
        print "In sample results"
        print "RMSE: ", rmseTrain
        cTrain = np.corrcoef(predYKNNTrain, y=trainY)
        print "corr: ", cTrain[0,1]

        learner = knn.KNNLearner(k = r) # constructor
        learner.addEvidence(trainX, trainY) # training step
        predYKNN = learner.query(testX) # query
        rmse = math.sqrt(((testY - predYKNN) ** 2).sum()/testY.shape[0])
        print
        print "KNN Learner"
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predYKNN, y=testY)
        print "corr: ", c[0,1]
        #end my test
        knnktest.append([r, rmseTrain, cTrain[0,1], rmse,c[0,1]])
    np.savetxt("Data/knnktest.csv", np.array(knnktest), delimiter=",")





    # create a learner and train it
    learner = lrl.LinRegLearner() # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "Linear Regression Learner"
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    blktest = []
    for r in range(1,101):
        import BagLearner as bl

        learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = r, boost = False)
        learner.addEvidence(trainX, trainY)
        predYBagTrain = learner.query(trainX)
        rmseTrain = math.sqrt(((trainY - predYBagTrain) ** 2).sum()/trainY.shape[0])
        print
        print "Bag Learner"
        print "Out of sample results"
        print "RMSE: ", rmseTrain
        cTrain = np.corrcoef(predYBagTrain, y=trainY)
        print "corr: ", cTrain[0,1]

        learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = r, boost = False)
        learner.addEvidence(trainX, trainY)
        predYBag = learner.query(testX)
        rmse = math.sqrt(((testY - predYBag) ** 2).sum()/testY.shape[0])
        print
        print "Bag Learner"
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predYBag, y=testY)
        print "corr: ", c[0,1]
        blktest.append([r, rmseTrain, cTrain[0,1], rmse,c[0,1]])
    np.savetxt("Data/blktest.csv", np.array(blktest), delimiter=",")
