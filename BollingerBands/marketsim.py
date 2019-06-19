"""MC2-P1: Market simulator."""
import datetime

import pandas as pd
import numpy as np
import os
import sys

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    #given df_trades, calculate df_value, df_portval
    def calculatePortval(df_trades, df_prices):
        #dataframe to represent how much of each stock we are holding and how much cash we have available each eay
        df_holdings = pd.DataFrame(0, index=df_prices.index, columns=df_prices.columns.values)
        #cash available. initialize to starting value.
        df_holdings.iloc[0]['CASH'] = start_val
        i = 0
        for index, row in df_trades.iterrows():
            #copy this row to next if there is next
            if i > 0:
                df_holdings.iloc[i] = df_holdings.iloc[i-1].values
            for header, value in row.iteritems():
                df_holdings.loc[index][header] += df_trades.loc[index][header]
            i += 1

        #value in dollars for each stock and cash same as df_holdingss
        df_value = df_holdings * df_prices

        #summ across columns to get your total portfolio worth stocks plus cash
        df_portval = df_value.sum(axis=1)

        return df_value, df_portval

    def leverage(row):
        #leverage = (sum(longs) + sum(abs(shorts))) / ((sum(long) - sum(abs(shorts)) + cash)
        sumLongs = 0
        sumShorts = 0
        for header, value in row.iteritems():
            if header != 'CASH':
                if value > 0.0:
                    sumLongs += value
                elif value < 0.0:
                    sumShorts += value
        sumAbsShorts = abs(sumShorts)
        return ((sumLongs + sumAbsShorts)/(sumLongs - sumAbsShorts + row['CASH']))


    def calculatePortvalWithLeverage(df_trades, df_prices, maxLev = 2):
        df_value, df_portval = calculatePortval(df_trades, df_prices)
        df_leverage = df_value.apply(leverage, axis=1)
        #dates with leverage over 2 Note this check with series only works with whole numbers.
        #make sure the increased leverage is because trade was made on that day not, because prices went up of existing stocks.
        levOver = df_leverage.loc[df_trades[df_trades['CASH'] != 0].index][df_leverage > maxLev].index

        if len(levOver) > 0:
            #erase all trades from that day
            df_trades.loc[levOver[0]] = 0
            return calculatePortvalWithLeverage(df_trades, df_prices, maxLev)
        return df_portval

    dates = pd.date_range(start_date, end_date)
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, usecols=['Date', 'Symbol', 'Order', 'Shares'], na_values=['nan'])
    symbols = pd.Series(orders['Symbol'].values.ravel()).unique().tolist()
    #get_data automatically adds SPY, so got to get only symbols
    df_prices = get_data(symbols, dates)[symbols]
    df_prices.loc[:,'CASH'] = pd.Series(1.0, index=df_prices.index)

    #get a dataframe with all 0s for stock tickers and cash for trading days
    df_trades = pd.DataFrame(0, index=df_prices.index, columns=df_prices.columns.values)

    #read orders one by one and add number of stocks bought at what price to df_trades
    for index, row in orders.iterrows():
        if row['Order'] == 'BUY':
            df_trades.loc[index][row['Symbol']] = df_trades.loc[index][row['Symbol']] + row['Shares']
            df_trades.loc[index]['CASH'] = df_trades.loc[index]['CASH'] - (row['Shares'] * df_prices.loc[index][row['Symbol']])
        else:
            df_trades.loc[index][row['Symbol']] = df_trades.loc[index][row['Symbol']] - row['Shares']
            df_trades.loc[index]['CASH'] = df_trades.loc[index]['CASH'] + (row['Shares'] * df_prices.loc[index][row['Symbol']])

    df_portval = calculatePortvalWithLeverage(df_trades, df_prices, 2)

    return df_portval


def test_run():
    #Driver function.
    # Define input parameters
    start_date = '2011-01-14'
    end_date = '2011-12-14'
    orders_file = os.path.join("orders", "orders2.csv")
    start_val = 1000000

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
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")

if __name__ == "__main__":
    sys.setrecursionlimit(999999999)
    test_run()
