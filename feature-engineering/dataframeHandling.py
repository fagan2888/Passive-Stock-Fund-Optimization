import pandas as pd
from sqlalchemy import create_engine
import datetime as dt

def get_dataframe_from_csv(file):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError as err:
        print("FileNotFoundError with path " + file + "\nError: " + err)
        raise
    return df


def add_columns_to_df(basic_df, columns):
    # Set Multi Index on the dataframe to get the 3d data structure
    try:
        new_df = basic_df.set_index(['Symbol', 'date_of_transaction'])
    except Exception as err:
        print("index set error")
        print(err)
        raise

    # Add columns to the new df
    for col in columns:
        new_df[col] = -1

    return new_df

def replace_table(df,table_name):
    engine = create_engine("postgresql://postgres:dfdk#418!@@35.237.73.115/postgres")
    #df=pd.read_csv(file_name)
    #df['Sno']=df.index
    #df['symbol']=df[' Symbol']
    #columns = ['Sno','date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close','symbol','Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed'] 
    df.to_sql(table_name,engine, if_exists='replace',index=False)

def read_table(table_name,max_date):
    engine = create_engine("postgresql://postgres:dfdk#418!@@35.237.73.115/postgres")
    #max_date2=dt.date(int(max_date.split('-')[0]),int(max_date.split('-')[1]),int(max_date.split('-')[2]))
    #sqlstr = ("'select * from ' + table_name +' where  date_of_transaction >' %s ", max_date2)
    
    df=pd.read_sql_query( 'select * from ' + table_name,con=engine)
    #print(df.head())
    #delta_records=df['date_of_transaction']>max_date
    return df



def find_max_date():

    engine = create_engine("postgresql://postgres:dfdk#418!@@35.237.73.115/postgres")
    df=pd.read_sql_query('select max("Date") as date_of_transaction from momentum_features',con=engine)
    last_pulled_date=df["date_of_transaction"].max()  
    return last_pulled_date

def load_table(df,table_name):
    engine = create_engine("postgresql://postgres:dfdk#418!@@35.237.73.115/postgres")
    #df=pd.read_csv(file_name)
    #df['sno']=df.index
    #df['symbol']=df[' Symbol']
    #columns = ['sno','date_of_transaction', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose','Symbol','Year','Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','Elapsed'] 
    df.to_sql(table_name,engine, if_exists='append',index=False)