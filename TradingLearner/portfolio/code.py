import pandas as pd
from bollinger_strategy import calculateBollingerBands, simulateOrders
from util import get_data
import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
import KNNLearner as knn
import math
import matplotlib.pyplot as plt


def plotTrain(df_all, title="Training"):
    plt.clf()
    df_all['Y'].plot(legend=True, color="green", label="Training Y")
    predPrice = ((df_all['Y Predicted'] + 1.0) * df_all['Stock'])
    predPrice.plot(legend=True, color="red", label="Predicted Y")
    df_all['Stock'].plot(legend=True, color="blue", label="Price")
    #ymin, ymax = ax.get_ylim()
    plt.legend(loc=3)
    plt.title(title)
    plt.show()


def generateSignal(df_all):
    """
    Creates a column with values to buy/sell.
    Buy 100 when price is predicted to go up 1% and sell after 5 days, unless after 5 days it is still predicted to go up.
    0: nothing
    1: enter long
    2: exit long
    3: enter short
    4: exit short
    """
    #add new column with all 0s
    df_all['Trade Signals'] = pd.Series(0, index=df_all.index)

    inPosition = False
    position = 0
    daysInPosition = 0
    for index, row in df_all.iterrows():
        if inPosition:
            daysInPosition = daysInPosition + 1
        #enter long position
        if (not inPosition) and (100*((((df_all.loc[index, 'Y Predicted'] + 1.0) * df_all.loc[index, 'Stock'])-df_all.loc[index, 'Stock'])/df_all.loc[index, 'Stock'])) > 1.0:
            df_all.loc[index, 'Trade Signals'] = 1
            inPosition = True
            position = 1
            daysInPosition = 0
        elif (inPosition) and (position == 1) and (daysInPosition > 5) and (100*((((df_all.loc[index, 'Y Predicted'] + 1.0) * df_all.loc[index, 'Stock'])-df_all.loc[index, 'Stock'])/df_all.loc[index, 'Stock'])) < 1.0:
            df_all.loc[index, 'Trade Signals'] = 2
            inPosition = False
            position = 2
            daysInPosition = 0
        elif (not inPosition) and (100*((((df_all.loc[index, 'Y Predicted'] + 1.0) * df_all.loc[index, 'Stock'])-df_all.loc[index, 'Stock'])/df_all.loc[index, 'Stock'])) < -1.0:
            df_all.loc[index, 'Trade Signals'] = 3
            inPosition = True
            position = 3
            daysInPosition = 0
        elif (inPosition) and (position == 3) and (daysInPosition > 5) and (100*((((df_all.loc[index, 'Y Predicted'] + 1.0) * df_all.loc[index, 'Stock'])-df_all.loc[index, 'Stock'])/df_all.loc[index, 'Stock'])) > -1.0:
            df_all.loc[index, 'Trade Signals'] = 4
            inPosition = False
            position = 4
            daysInPosition = 0
    return df_all


def plotEntryExits(df_all, title="Trading Signals"):
    plt.clf()
    #df_all['Y'].plot(legend=True, color="green", label="Training Y")
    predY = ((df_all['Y Predicted'] + 1.0) * df_all['Stock'])
    predY.plot(legend=True, color="red", label="Predicted Y")
    ax = df_all['Stock'].plot(legend=True, color="blue", label="Price")
    ymin, ymax = ax.get_ylim()

    #get entry exit values
    df_trades_enter_long = df_all.loc[df_all[df_all['Trade Signals'] == 1].index, 'Trade Signals']
    df_trades_exit_long = df_all.loc[df_all[df_all['Trade Signals'] == 2].index, 'Trade Signals']
    df_trades_enter_short = df_all.loc[df_all[df_all['Trade Signals'] == 3].index, 'Trade Signals']
    df_trades_exit_short = df_all.loc[df_all[df_all['Trade Signals'] == 4].index, 'Trade Signals']

    plt.vlines(df_trades_enter_long.index, ymin, ymax, colors='green')
    plt.vlines(df_trades_exit_long.index, ymin, ymax, colors='black')
    plt.vlines(df_trades_enter_short.index, ymin, ymax, colors='red')
    plt.vlines(df_trades_exit_short.index, ymin, ymax, colors='black')

    plt.legend(loc=3)
    plt.title(title)
    plt.show()

def generateOrders(stock_symbol, df_all, file_name):
    series_trades = df_all.loc[df_all[df_all['Trade Signals'] != 0].index, 'Trade Signals']
    df_trades = pd.DataFrame(series_trades, series_trades.index, columns={'Trade Signals'}).rename(columns={'Trade Signals':'Order'})
    df_trades = df_trades.replace([1,2,3,4],['BUY','SELL','SELL','BUY'])
    #add new column with number of shares as 100
    df_trades['Shares'] = pd.Series(100, index=df_trades.index)
    df_trades['Symbol'] = pd.Series(stock_symbol, index=df_trades.index)
    df_trades.to_csv(file_name, index_label='Date')

if __name__ == "__main__":
    n = 10 #for rolling std
    num_pred_days = 5
    k = 5 #for knn learner
    num_bags = 10

    start_val = 10000
    orders_file = 'orders.txt'
    start_date = '2008-01-01'
    end_date = '2009-12-31'
    #symbol = 'ML4T-399'
    symbol = 'IBM'
    dates = pd.date_range(start_date, end_date)

    #df_all is dataframe containing stock price, sma, upper b band, lower b band
    df_all = calculateBollingerBands(symbol, start_date, end_date, n)
    bb_value = ((df_all['Stock'] - df_all['SMA'])/(df_all['Upper Bollinger Band'] - df_all['SMA']))
    bb_value = bb_value/bb_value.loc[bb_value.abs().idxmax()].astype(np.float64) #normalize
    bb_value.name = 'Bollinger Value'
    momentum = (df_all['Stock']/df_all['Stock'].shift(n)) - 1
    momentum = momentum/momentum.loc[momentum.abs().idxmax()].astype(np.float64) #normalize
    momentum.name = 'Momentum'
    #volatility = pd.rolling_std(((df_all['Stock']/df_all['Stock'].shift(1))-1), n, min_periods=n)
    volatility = ((pd.rolling_mean(df_all['Stock'], n, min_periods=n) - df_all['Stock'])/ pd.rolling_std(df_all['Stock'], n, min_periods=n))
    volatility = volatility/volatility.loc[volatility.abs().idxmax()].astype(np.float64) #normalize
    volatility.name = 'Volatility'
    y_normed = (df_all['Stock'].shift(-num_pred_days)/df_all['Stock']) - 1.0 #small values
    y_normed.name = 'Y Normed'
    y = (df_all['Stock']*(y_normed + 1.0)) #what we think the actual price will be after 5 days
    y.name = 'Y'

    #df_all = pd.concat([df_all, bb_value, momentum, volatility, y_normed, y], axis=1)

    #df_training = df_all[['Bollinger Value','Momentum', 'Volatility', 'Y']][n:df_all.shape[0]-num_pred_days]
    df_training = pd.concat([bb_value, momentum, volatility, y_normed], axis=1)
    df_training = df_training[n:df_training.shape[0]-num_pred_days]

    training_data = df_training.as_matrix(columns=df_training.columns[:])
    trainX = training_data[:,0:-1]
    trainY = training_data[:,-1]
    #learner = bl.BagLearner(learner=knn.KNNLearner, kwargs={"k":k},bags=num_bags)
    learner = knn.KNNLearner(k=k)
    #learner = lrl.LinRegLearner()
    learner.addEvidence(trainX, trainY)
    predY = learner.query(trainX)

    predY = pd.Series(predY, index=df_training.index)
    predY.name = 'Y Predicted'

    """
    rmseTrain = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print "RMSE: ", rmseTrain
    cTrain = np.corrcoef(predY, y=trainY)
    print "corr: ", cTrain[0,1]
    """

    df_all = pd.concat([df_all, bb_value, momentum, volatility, y_normed, y, predY], axis=1)

    plotTrain(df_all, title=str(symbol) + " - Training")

    #generate Buy Sell signal.
    df_all = generateSignal(df_all)
    plotEntryExits(df_all, title=str(symbol) + " - Training Trading Signals")

    #generate order file
    generateOrders(symbol, df_all, orders_file)
    simulateOrders(start_date, end_date, orders_file, start_val, title=str(symbol) + " - Training Portfolio Value")


    #Repeat all steps for testing data
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    dates = pd.date_range(start_date, end_date)

    df_all = calculateBollingerBands(symbol, start_date, end_date, n)
    bb_value = ((df_all['Stock'] - df_all['SMA'])/(df_all['Upper Bollinger Band'] - df_all['SMA']))
    bb_value = bb_value/bb_value.loc[bb_value.abs().idxmax()].astype(np.float64) #normalize
    bb_value.name = 'Bollinger Value'
    momentum = (df_all['Stock']/df_all['Stock'].shift(n)) - 1
    momentum = momentum/momentum.loc[momentum.abs().idxmax()].astype(np.float64) #normalize
    momentum.name = 'Momentum'
    volatility = ((pd.rolling_mean(df_all['Stock'], n, min_periods=n) - df_all['Stock'])/ pd.rolling_std(df_all['Stock'], n, min_periods=n))
    volatility = volatility/volatility.loc[volatility.abs().idxmax()].astype(np.float64) #normalize
    volatility.name = 'Volatility'
    y_normed = (df_all['Stock'].shift(-num_pred_days)/df_all['Stock']) - 1.0 #small values
    y_normed.name = 'Y Normed'
    y = (df_all['Stock']*(y_normed + 1.0)) #what we think the actual price will be after 5 days
    y.name = 'Y'

    df_testing = pd.concat([bb_value, momentum, volatility, y_normed], axis=1)
    df_testing = df_testing[n:df_testing.shape[0]-num_pred_days]

    testing_data = df_testing.as_matrix(columns=df_testing.columns[:])
    testX = testing_data[:,0:-1]
    testY = testing_data[:,-1]
    predY = learner.query(testX)

    predY = pd.Series(predY, index=df_testing.index)
    predY.name = 'Y Predicted'

    df_all = pd.concat([df_all, bb_value, momentum, volatility, y_normed, y, predY], axis=1)

    plotTrain(df_all, title=str(symbol) + " - Testing")

    #generate Buy Sell signal.
    df_all = generateSignal(df_all)
    plotEntryExits(df_all, title=str(symbol) + " - Testing Trading Signals")

    #generate order file
    generateOrders(symbol, df_all, orders_file)
    simulateOrders(start_date, end_date, orders_file, start_val, title=str(symbol) + " - Testing Portfolio Value")

    print ""

