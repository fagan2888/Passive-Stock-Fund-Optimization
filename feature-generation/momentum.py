import pandas as pd
from talib import RSI, BBANDS
import matplotlib.pyplot as plt
import time


START_DATE = '2011-01-03'
END_DATE = '2019-04-03'

def get_stock_rsi_daily():
    pass


def get_stock_percent_off_52_week_high():
    pass


def get_stock_trailing_52_week_performance_rank():
    pass


def get_stock_volatility():
    pass


def get_stock_sharp_ratio():
    pass


def main():

    print('Generating Momentum Features\n-------------------------------------------------------------')
    file_path = "../data/stock_price_until_2019_04_03.csv"
    output_file_path = "../data/momentum-features.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as err:
        print("FileNotFoundError with path " + file_path + "\nError: " + err)
        raise

    print('converting 2d to 3d dataframe...........')

    # Set Multi Index on the dataframe to get the 3d data structure
    df.set_index(['Symbol','Date'], inplace=True)
    print('Done')
    # Example lookup
    #print(df.loc['AAPL'].loc[:,'AdjClose'].tail())

    # Add Columns to the dataframe and initialize to -1
    df['RSI'] = -1
    df['Volatility'] = -1


    close = df.loc['AAPL'].loc[::,'AdjClose'].values
    rsi = RSI(close, timeperiod=20)
    rsi_series = pd.Series(rsi)

    print('Updating Dataframe with RSI values......')
    start = time.time()
    print(df.loc['AAPL']['RSI'])
    for symbol, mrow in df.groupby(level=0):
        counter = 0
        print(symbol)
        close = df.loc[symbol].loc[::, 'AdjClose'].values
        rsi = RSI(close, timeperiod=20)
        rsi_series = pd.Series(rsi)
        tmpDf = pd.DataFrame(data=rsi_series, columns=['RSI'])
        df.loc[symbol, 'RSI'] = tmpDf['RSI'].values
        """
        for index, row in df.loc[symbol[0]].iterrows():
            try:
                df.loc[(symbol, index), 'RSI'] = rsi_series[counter]
            except Exception as err:
                print("exception: " + str(err))
                raise
            counter = counter + 1
        print(df.loc[symbol, 'RSI'])
        """

    end = time.time()
    print("Process time: " + str(end - start) + " seconds.")
    df.to_csv(output_file_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
