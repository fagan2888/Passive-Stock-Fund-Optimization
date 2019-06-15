import pandas as pd
import datetime as dt
from pandas_datareader import data
import os
import contextlib
import time
import PIL, os, numpy as np, math, collections, threading, json,  random, scipy
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
import os.path
from os import path
import sqlite3
from sqlalchemy import create_engine





def pull_spy_list():
    spylist = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = spylist[0]

    #Saving SPY List to a csv file
    table.to_csv(r'Data/spy_list.csv', header=True, index=None, mode='a', sep=',')
    df=pd.read_csv('Data/spy_list.csv')
    return df

def load_table(file_name,table_name):
    engine = create_engine("postgresql://postgres:dfdk#418!@@35.237.73.115/postgres")
    df=pd.read_csv(file_name)
    df['Sno']=df.index
    df['symbol']=df[' Symbol']
    columns = ['Sno','date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close','symbol','Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed'] 
    df[columns].to_sql(table_name,engine, if_exists='append',index=False)

def save_to_database():
    DBPATH='Data/capstone.db'
    cnx = sqlite3.connect(DBPATH)
    df = pd.read_csv('Data/stock_price.csv')
    df.to_sql('stock_price', cnx)   

def find_max_date():
    engine = create_engine("postgresql://postgres:dfdk#418!@@35.237.73.115/postgres")
    df=pd.read_sql_query('select max(date_of_transaction) as date_of_transaction from spy_stock_price',con=engine)
    last_pulled_date=df['date_of_transaction'].max()

    #exists = os.path.isfile('Data/spy_stock_price.csv')

    #if exists:
    #    df = pd.read_csv('Data/spy_stock_price.csv')
    #    last_pulled_date=df['date_of_transaction'].max()
    #else:
    #    last_pulled_date=dt.date(2011, 1, 1)
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

    
    
    count=0
    last_pulled_date=find_max_date()


    #Increasing to next date
    last_pulled_date2=datetime.date(int(last_pulled_date.split('-')[0]),int(last_pulled_date.split('-')[1]),int(last_pulled_date.split('-')[2]))+ datetime.timedelta(days=1)

    #print (last_pulled_date2)
    exists = os.path.isfile('Data/basic_spy_stock_price.csv')

    

    if exists:
        os.remove('Data/basic_spy_stock_price.csv')
  
    lst=["SPY"]

    for v in lst:
        stock = data.DataReader(name=v.replace('.','-'),data_source="yahoo", start=last_pulled_date2, end=dt.datetime.now())
        stock['Symbol']=v
        exists = os.path.isfile('Data/basic_spy_stock_price.csv')

        if exists:
            
            stock.to_csv(r'Data/basic_spy_stock_price.csv', header=False, index=True, mode='a', sep=',')
        else:
            
            stock.to_csv(r'Data/basic_spy_stock_price.csv', header=True, index=True, mode='a', sep=',')
        count=count+1

    df = pd.read_csv('Data/basic_spy_stock_price.csv')
    print( df['Date'].min(), df['Date'].max())
    df.columns = ['date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',' Symbol']
    print(df.head())
    add_datepart(df, 'date_of_transaction')
    df.columns = ['date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',' Symbol','Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed'] 
    df.reset_index()
    df.columns = ['date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',' Symbol','Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed'] 
    
    print(df.head())

    exists = os.path.isfile('Data/spy_stock_price.csv')

    if exists:
        os.remove('Data/spy_stock_price.csv')
    

    df.to_csv(r'Data/spy_stock_price.csv', header=True, index=True, mode='a', sep=',')
    
    #Saving to sqlite
    load_table('Data/spy_stock_price.csv','spy_stock_price')

##############################
##Execution
##############################
if __name__ == '__main__':
    main()
