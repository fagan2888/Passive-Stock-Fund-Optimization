"""Stock feature generation including momentum indicators and volatility

    Currently RSI (Relative Strength Index) is successfully calculated for each stock
    and for each day based on the basic price data ingested from yahoo. To Run the program,
    the stock_price_until_2019_04_03.csv file must be located in the same directory as this
    file. The output of the prgram will be an updated csv file with a new column with the
    rsi values.

    TODO: volatility, percent off 52 week high, stock relative rank (based on price performance), and Sharp Ratio

George Krug
04/22/2019
"""



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
    file_path = "stock_price_until_2019_04_03.csv"
    output_file_path = "momentum-features.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as err:
        print("FileNotFoundError with path " + file_path + "\nError: " + err)
        raise

    # Set Multi Index on the dataframe to get the 3d data structure
    df.set_index(['Symbol','Date'], inplace=True)
    print('Done')

    # Add Columns to the dataframe and initialize to -1
    df['RSI'] = -1
    df['Volatility'] = -1


    close = df.loc['AAPL'].loc[::,'AdjClose'].values
    rsi = RSI(close, timeperiod=20)
    rsi_series = pd.Series(rsi)

    print('Updating Dataframe with RSI values......')
    start = time.time()
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

    print("Writing to file: " + output_file_path)
    df.to_csv(output_file_path, encoding='utf-8', index=False)
    end = time.time()
    print("Process time: " + str(end - start) + " seconds.")


if __name__ == '__main__':
    main()
