"""
Generate daily, monthly, and yearly rolling stock returns ((Ending Value - Begining Value) / (Ending Value)) based on the basic price data ingestet
from the yahoo finance api. To run this program, the stock_price_until_2019_04_03.csv must be
in the same working directory.

George Krug
04/25/19
"""


import pandas as pd
from taIndicators import basic
import dataframeHandling as dfhandle
import time


def get_daily_percent_change(time_series_df, ticker):
    close = pd.DataFrame(basic.get_time_series_adjusted_close(time_series_df, ticker))
    time_series_df.loc[ticker, 'Pct_Change_Daily'] = close.pct_change(1).values
    time_series_df.loc[ticker, 'Pct_Change_Monthly'] = close.pct_change(basic.MONTHLY_TRADING_DAYS).values
    time_series_df.loc[ticker, 'Pct_Change_Yearly'] = close.pct_change(basic.YEARLY_TRADING_DAYS).values
    return time_series_df


def get_stock_return_features():
    file_path = "data/stock_price_until_2019_04_03.csv"
    output_file_path = "data/stock_prices_and_returns.csv"
    new_columns = ['Pct_Change_Daily', 'Pct_Change_Monthly', 'Pct_Change_Yearly']
    basic_df = dfhandle.get_dataframe_from_csv(file_path)
    df = dfhandle.add_columns_to_df(basic_df, new_columns)

    start = time.time()
    print("Calculating price return data....................")
    for symbol, mrow in df.groupby(level=0):
        print(symbol)
        df = get_daily_percent_change(df, symbol)


    print("Writing to file: " + output_file_path)
    end = time.time()
    df.to_csv(output_file_path, encoding='utf-8', index=True)
    print("Process time: " + str(end - start) + " seconds.")

get_stock_return_features()

