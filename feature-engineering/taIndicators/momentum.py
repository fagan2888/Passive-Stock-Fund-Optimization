"""
Momentum module containing methods to generate momentum features
including RSI and rolling price return rank (monthly and yearly)
"""


import pandas as pd
from talib import RSI
import numpy as np
import matplotlib.pyplot as plt
import math


START_DATE = '2011-01-03'
END_DATE = '2019-04-03'
RANK_RECALCULATE = 1
YEARLY_TRADING_DAYS = 252
MONTHLY_TRADING_DAYS = 21


def get_stock_rsi_daily(time_series_df, ticker):
    """
    compute rolling RSI of stock prices using talib
    """
    close = get_time_series_adjusted_close(time_series_df, ticker)
    rsi = RSI(close, timeperiod=20)
    rsi_series = pd.Series(rsi)
    tmpDf = pd.DataFrame(data=rsi_series, columns=['RSI'])
    time_series_df.loc[ticker, 'RSI'] = tmpDf['RSI'].values
    return time_series_df


def get_stock_percent_off_52_week_high():
    pass


def update_rank_dataframe(df, stock_returns, period, date):
    """
    For each stock trading day in the data range, update the
    rolling return rank based on the new monthly & yearly
    percent change value
    """
    stock_period = str(period) + "_Return"
    rank_period = str(period) + "_Return_Rank"
    df_tmp = df.reset_index(level=0)
    returns_df = pd.DataFrame.from_dict(stock_returns, orient='index', columns=[stock_period])
    returns_df.sort_values(by=[stock_period], ascending=False, inplace=True)
    returns_df.reset_index(level=0, inplace=True)
    returns_df[period] = returns_df.index
    returns_df.columns = ['Symbol', stock_period, rank_period]
    daily_adjusted_rank_df = pd.DataFrame()
    daily_adjusted_rank_df = pd.merge(df_tmp.loc[date], returns_df, on='Symbol', how='left')
    daily_adjusted_rank_df['Date'] = date
    daily_adjusted_rank_df.set_index(['Symbol', 'Date'], inplace=True)

    return daily_adjusted_rank_df


def update_with_null_return_rankings(df, stocks_dict, period, date):
    """
    Since the early dates do not have enough data to compute monthly/yearly
    percent chages, copy the data frame values
    """
    stock_period = str(period) + "_Return"
    rank_period = str(period) + "_Return_Rank"
    df_tmp = df.reset_index(level=0)
    daily_rank_df = pd.DataFrame.from_dict(stocks_dict, orient='index', columns=[stock_period])
    daily_rank_df.reset_index(level=0, inplace=True)
    daily_rank_df[rank_period] = np.nan
    daily_rank_df.columns = ['Symbol', stock_period, rank_period]
    updated_daily_subset_df = pd.DataFrame()
    updated_daily_subset_df = pd.merge(df_tmp.loc[date], daily_rank_df, on='Symbol', how='left')
    updated_daily_subset_df['Date'] = date
    updated_daily_subset_df.set_index(['Symbol', 'Date'], inplace=True)
    return updated_daily_subset_df


def get_daily_adjusted_stock_return_rankings(df, ticker_list, date_list):
    """
    The input df dataframe must contain monthly & yearly stock percent changes
    to compute a rolling return rank updated daily
    """
    global yearl_rank_df, monthly_rank_df
    yearly_rank_df = pd.DataFrame()
    monthly_rank_df = pd.DataFrame()

    for date in date_list:
        yearly = {}
        monthly = {}
        yearly_no_data = {}
        monthly_no_data = {}
        print(date)

        for symbol in ticker_list:
            try:
                if math.isnan(df.loc[symbol, date].loc['Pct_Change_Yearly']):
                    yearly_no_data[symbol] = 'x'
                if not math.isnan(df.loc[symbol, date].loc['Pct_Change_Yearly']):
                    yearly[symbol] = df.loc[symbol, date].loc['Pct_Change_Yearly']
            except Exception as err:
                yearly_no_data[symbol] = 'x'

            try:
                if math.isnan(df.loc[symbol, date].loc['Pct_Change_Monthly']):
                    monthly_no_data[symbol] = 'x'
                if not math.isnan(df.loc[symbol, date].loc['Pct_Change_Monthly']):
                    monthly[symbol] = df.loc[symbol, date].loc['Pct_Change_Monthly']
            except Exception as err:
                monthly_no_data[symbol] = 'x'

        if len(yearly) > 0:
            daily_adjusted_rank_df = update_rank_dataframe(df, yearly, "Yearly", date)
            yearly_rank_df = yearly_rank_df.append(daily_adjusted_rank_df)
        else:
            new_df = update_with_null_return_rankings(df, yearly_no_data, "Yearly", date)
            yearly_rank_df = yearly_rank_df.append(new_df)
        if len(monthly) > 0:
            daily_adjusted_rank_df = update_rank_dataframe(df, monthly, "Monthly", date)
            monthly_rank_df = monthly_rank_df.append(daily_adjusted_rank_df)
        else:
            new_df = update_with_null_return_rankings(df, monthly_no_data, "Monthly", date)
            monthly_rank_df = monthly_rank_df.append(new_df)

    yearly_rank_df.reset_index(inplace=True)
    monthly_rank_df.reset_index(inplace=True)
    return yearly_rank_df, monthly_rank_df


def get_time_series_adjusted_close(time_series_df, ticker):
    return time_series_df.loc[ticker].loc[::, 'AdjClose'].values


def get_percent_positive_days(df, period):
    pass


def get_max_daily_return(df, period):
    pass