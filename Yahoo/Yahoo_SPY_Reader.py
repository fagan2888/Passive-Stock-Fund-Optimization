import pandas as pd
import datetime as dt
from pandas_datareader import data
import os
import contextlib
import time



@contextlib.contextmanager
def atomic_overwrite(filename):
    temp = filename + '~'
    with open(temp, "w") as f:
        yield f
    os.rename(temp, filename) # this will only happen if no exception was raised

def save_to_csv_without_header(filename,df):
    print(df.head(5))

#def gather_spy_list():


def pull_spy_list():
    spylist = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = spylist[0]

    #Saving SPY List to a csv file
    with atomic_overwrite("spy_list.csv") as f:
        table.to_csv(r'spy_list.csv', header=False, index=None, mode='a', sep=',')
    df=pd.read_csv('spy_list.csv')
    return df

def save_to_database():
    DBPATH='/Users/syandra/Documents/sqlite-tools-osx-x86-3270200/capstone.db'
    cnx = sqlite3.connect(DBPATH)
    df = pandas.read_csv('stock_price.csv')
    df.to_sql('stock_price2', cnx)   


def main():

    #df=pull_spy_list()
    #Pull the list of S&P Stocks
    spylist = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = spylist[0]

    #Saving SPY List to a csv file
    #with atomic_overwrite("spy_list.csv") as f:
    table.to_csv(r'spy_list.csv', header=False, index=None, mode='a', sep=',')
    df=pd.read_csv('spy_list.csv')

    
    
    
    count=0
    for v in df['Symbol']:
        stock = data.DataReader(name=v.replace('.','-'),data_source="yahoo", start=dt.date(2011, 1, 1), end=dt.datetime.now())
        stock['Symbol']=v
        if count==0:
            with atomic_overwrite("stock_price.csv") as f:
                pass
                stock.to_csv(r'stock_price.csv', header=True, index=True, mode='a', sep=',')
        else:
            pass
            stock.to_csv(r'stock_price.csv', header=False, index=True, mode='a', sep=',')
        count=count+1
        
    save_to_database()

##############################
##Execution
##############################
if __name__ == '__main__':
    main()
