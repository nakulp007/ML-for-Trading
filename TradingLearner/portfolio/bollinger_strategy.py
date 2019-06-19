import pandas as pd
import matplotlib.pyplot as plt

from analysis import get_portfolio_stats, get_portfolio_value, plot_normalized_data
from marketsim import compute_portvals
from portfolio.util import get_data


def calculateBollingerBands(stock_symbol, start_date, end_date, n=20):
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([stock_symbol], dates)  # automatically adds SPY
    df_stock_price = prices_all[[stock_symbol]]  # only my symbols
    price_SPY = prices_all['SPY']  # only SPY, for comparison later

    #n-day simple moving average
    df_SMA = pd.rolling_mean(df_stock_price, n, min_periods=n).rename(columns={stock_symbol: 'SMA'})

    #moving standard deviation
    df_moving_std = pd.rolling_std(df_stock_price, n, min_periods=n).rename(columns={stock_symbol: 'MSTD'})
    #calculate bollinger bands
    bollinger_upperband = (df_SMA['SMA'] + (2*df_moving_std['MSTD']))
    bollinger_lowerband = df_SMA['SMA'] - (2*df_moving_std['MSTD'])
    df_bollinger_bands = pd.concat([bollinger_upperband, bollinger_lowerband], axis=1).rename(columns={0: 'Upper Bollinger Band', 1: 'Lower Bollinger Band'})
    #.apply(tuple, axis=1)#
    #df_bollinger_bands['Bollinger Bands'] = df_bollinger_bands[['Upper Bollinger Band', 'Lower Bollinger Band']].apply(tuple, axis=1)
    prices_all.rename(columns={stock_symbol: 'Stock'}, inplace=True)
    df_temp = pd.concat([prices_all, df_SMA, df_bollinger_bands], axis=1)
    #df_temp2 = pd.DataFrame(df_temp,index=df_temp.index, columns=['IBM', 'SMA', 'Upper Bollinger Band', 'Lower Bollinger Band'])
    #plot_data(df_temp)
    return df_temp


def computeTrades(df_all):
    """
    0: nothing
    1: enter long
    2: exit long
    3: enter short
    4: exit short
    """
    #add new column with all 0s
    df_all['Bollinger Trades'] = pd.Series(0, index=df_all.index)

    inPosition = False
    prev_index = -1
    for index, row in df_all.iterrows():
        #make sure this is not first value in dataframe. we cant look at previous value for first value
        if prev_index == -1:
            prev_index = index
        else:
            #check if entering from under bollinger band, enter long position
            if (not inPosition) and (df_all.loc[prev_index,'Stock'] < df_all.loc[prev_index,'Lower Bollinger Band']) and (df_all.loc[index,'Stock'] > df_all.loc[index,'Lower Bollinger Band']) and (df_all.loc[index,'Stock'] < df_all.loc[index,'SMA']):
                df_all.loc[index, 'Bollinger Trades'] = 1
                inPosition = True
            #check if time to exit long position
            elif (inPosition) and (df_all.loc[prev_index,'Stock'] < df_all.loc[prev_index,'SMA']) and (df_all.loc[index,'Stock'] > df_all.loc[index,'SMA']):
                df_all.loc[index, 'Bollinger Trades'] = 2
                inPosition = False
            #check if entering from above bollinger band, enter short position
            elif (not inPosition) and (df_all.loc[prev_index,'Stock'] > df_all.loc[prev_index,'Upper Bollinger Band']) and (df_all.loc[index,'Stock'] < df_all.loc[index,'Upper Bollinger Band']) and (df_all.loc[index,'Stock'] > df_all.loc[index,'SMA']):
                df_all.loc[index, 'Bollinger Trades'] = 3
                inPosition = True
            #check if time to exit short position
            elif (inPosition) and (df_all.loc[prev_index,'Stock'] > df_all.loc[prev_index,'SMA']) and (df_all.loc[index,'Stock'] < df_all.loc[index,'SMA']):
                df_all.loc[index, 'Bollinger Trades'] = 4
                inPosition = False
            prev_index = index
    return df_all


def plotAll(stock_symbol, df_all):
    plt.clf()
    df_all['Stock'].plot(legend=True, color="blue", label=stock_symbol)
    df_all['SMA'].plot(legend=True, color="yellow")
    df_all['Upper Bollinger Band'].plot(color="cyan", label="Bollinger Bands")
    ax = df_all['Lower Bollinger Band'].plot(color="cyan", label='')

    #get entry exit values
    df_bollinger_trades_enter_long = df_all.loc[df_all[df_all['Bollinger Trades'] == 1].index, 'Bollinger Trades']
    df_bollinger_trades_exit_long = df_all.loc[df_all[df_all['Bollinger Trades'] == 2].index, 'Bollinger Trades']
    df_bollinger_trades_enter_short = df_all.loc[df_all[df_all['Bollinger Trades'] == 3].index, 'Bollinger Trades']
    df_bollinger_trades_exit_short = df_all.loc[df_all[df_all['Bollinger Trades'] == 4].index, 'Bollinger Trades']

    ymin, ymax = ax.get_ylim()
    plt.vlines(df_bollinger_trades_enter_long.index, ymin, ymax, colors='green')
    plt.vlines(df_bollinger_trades_exit_long.index, ymin, ymax, colors='black')
    plt.vlines(df_bollinger_trades_enter_short.index, ymin, ymax, colors='red')
    plt.vlines(df_bollinger_trades_exit_short.index, ymin, ymax, colors='black')
    plt.legend(loc=3)
    plt.show()


def generateOrders(stock_symbol, df_all, file_name):
    series_bollinger_trades = df_all.loc[df_all[df_all['Bollinger Trades'] != 0].index, 'Bollinger Trades']
    df_bollinger_trades = pd.DataFrame(series_bollinger_trades, series_bollinger_trades.index, columns={'Bollinger Trades'}).rename(columns={'Bollinger Trades':'Order'})
    df_bollinger_trades = df_bollinger_trades.replace([1,2,3,4],['BUY','SELL','SELL','BUY'])
    #add new column with number of shares as 100
    df_bollinger_trades['Shares'] = pd.Series(100, index=df_bollinger_trades.index)
    df_bollinger_trades['Symbol'] = pd.Series(stock_symbol, index=df_bollinger_trades.index)
    df_bollinger_trades.to_csv(file_name, index_label='Date')


def simulateOrders(start_date, end_date, orders_file, start_val, title="Portfolio Value"):
    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_temp, title)


def run():
    # Define input parameters
    stock_symbol = 'IBM'
    start_date = '2007-12-31'
    end_date = '2009-12-31'
    start_val = 10000
    orders_file = 'orders.txt'

    #SPY, IBM, SMA, Upper Bollinger Band, Lower Bollinger Band
    df_all = calculateBollingerBands(stock_symbol, start_date, end_date)
    df_all = computeTrades(df_all)

    generateOrders(stock_symbol, df_all, orders_file)
    simulateOrders(start_date, end_date, orders_file, start_val)
    plotAll(stock_symbol, df_all)




if __name__ == "__main__":
    run()