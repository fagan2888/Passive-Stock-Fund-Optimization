import pandas as pd
import datetime as dt
from pandas_datareader import data
import os
import contextlib
import time
import PIL, os, numpy as np, math, collections, threading, json,  random, scipy, cv2
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
import os.path
from os import path






def pull_spy_list():
    spylist = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = spylist[0]

    #Saving SPY List to a csv file
    table.to_csv(r'spy_list.csv', header=True, index=None, mode='a', sep=',')
    df=pd.read_csv('spy_list.csv')
    return df

def save_to_database():
    DBPATH='/Users/syandra/Documents/sqlite-tools-osx-x86-3270200/capstone.db'
    cnx = sqlite3.connect(DBPATH)
    df = pandas.read_csv('stock_price.csv')
    df.to_sql('stock_price2', cnx)   

def find_max_date():
    exists = os.path.isfile('stock_price.csv')

    if exists:
        df = pd.read_csv('stock_price.csv')
        last_pulled_date=df['date_of_transaction'].max()
    else:
        last_pulled_date=dt.date(2011, 1, 1)
    return last_pulled_date


def add_datepart(df, fldname, drop=True, time=False, errors="raise"):	
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> add_datepart(df, 'A')
    >>> df
        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    #if drop: df.drop(fldname, axis=1, inplace=True)



def main():

    
    #Pull the list of S&P Stocks
    
    

    #Saving SPY List to a csv file
    #with atomic_overwrite("spy_list.csv") as f:
    exists = os.path.isfile('spy_list.csv')
    if exists:
        pass
    else:
        spylist = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        table = spylist[0]
        table.to_csv(r'spy_list.csv', header=True, index=None, mode='a', sep=',')
    
    
    df=pd.read_csv('spy_list.csv')

    
    
    
    count=0
    last_pulled_date=find_max_date()

    for v in df['Symbol']:
        stock = data.DataReader(name=v.replace('.','-'),data_source="yahoo", start=last_pulled_date, end=dt.datetime.now())
        stock['Symbol']=v
        exists = os.path.isfile('basic_stock_price.csv')
        if exists:
            
            stock.to_csv(r'basic_stock_price.csv', header=False, index=True, mode='a', sep=',')
        else:
            
            stock.to_csv(r'basic_stock_price.csv', header=True, index=True, mode='a', sep=',')
        count=count+1

    df = pd.read_csv('basic_stock_price.csv')
    print( df['Date'].min(), df['Date'].max())
    df.columns = ['date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',' Symbol']
    print(df.head())
    add_datepart(df, 'date_of_transaction')
    df.columns = ['date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',' Symbol','Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed'] 
    
    print(df.head())

    df.to_csv(r'stock_price.csv', header=True, index=True, mode='a', sep=',')
    #save_to_database()

##############################
##Execution
##############################
if __name__ == '__main__':
    main()
