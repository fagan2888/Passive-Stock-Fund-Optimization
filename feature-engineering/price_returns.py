"""
Generate daily, monthly, and yearly rolling stock returns ((Ending Value - Begining Value) / (Ending Value)) based on the basic price data ingestet
from the yahoo finance api. To run this program, the stock_price_until_2019_04_03.csv must be
in the same working directory.

George Krug
04/25/19
"""


import pandas as pd
from talib import RSI
import matplotlib.pyplot as plt
import time
#import momentum
from taIndicators import momentum
import dataframeHandling as dfhandle


  

def get_daily_percent_change(time_series_df, ticker):
    close = pd.DataFrame(momentum.get_time_series_adjusted_close(time_series_df, ticker))
    time_series_df.loc[ticker, 'Pct_Change_Daily'] = close.pct_change(1).values
    time_series_df.loc[ticker, 'Pct_Change_Monthly'] = close.pct_change(momentum.MONTHLY_TRADING_DAYS).values
    time_series_df.loc[ticker, 'Pct_Change_Yearly'] = close.pct_change(momentum.YEARLY_TRADING_DAYS).values
    return time_series_df


def get_stock_return_features():
    file_path = "data/stock_price.csv"
    output_file_path = "data/stock_prices_and_returns_2.csv"
    
    bacis_columns=['sno', 'date_of_transaction','High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose','Symbol','ticker','Date']
    new_columns = ['Pct_Change_Daily', 'Pct_Change_Monthly', 'Pct_Change_Yearly']
    #basic_df = dfhandle.get_dataframe_from_csv(file_path)
    max_date=dfhandle.find_max_date()
    print(max_date)
    start = time.time()

    basic_df=dfhandle.read_table('stock_price',max_date)
    basic_df['Date']=basic_df['date_of_transaction']
    end = time.time()
    print("Time for reading table: " + str(end - start) + " seconds.")
    print(basic_df.head())
    basic_df['ticker']=basic_df['Symbol']
    
    df = dfhandle.add_columns_to_df(basic_df[bacis_columns], new_columns)

    start = time.time()
    
    print("Calculating price return data....................")
    for symbol, mrow in df.groupby(level=0):
        print(symbol)
        df = get_daily_percent_change(df, symbol)
        


    print("Writing to file: " + output_file_path)
    
    print(df.columns)
    df['Symbol']=df['ticker']
    df['date_of_transaction']=df['Date']
    print(df.columns)
    #new_df = df['date_of_transaction'] > '2019-04-03'
    print(df.info())
    df.columns=['sno','High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose','ticker','Date','Pct_Change_Daily', 'Pct_Change_Monthly', 'Pct_Change_Yearly','Symbol','date_of_transaction']

    delta_records=df['date_of_transaction']>max_date

    dfhandle.replace_table(df[delta_records],'stock_price_return')

    end = time.time()
    
    #df.to_csv(output_file_path, encoding='utf-8', index=True)
    print("Process time: " + str(end - start) + " seconds.")

get_stock_return_features()