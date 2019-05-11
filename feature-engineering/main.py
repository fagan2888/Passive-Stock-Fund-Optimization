import pandas as pd
import time
from taIndicators import momentum, volatility as vol
import sys


def get_dataframe_from_csv(file):
    """
    Open file; raise exception if it does not exist
    """
    try:
        df = pd.read_csv(file)
    except FileNotFoundError as err:
        print("FileNotFoundError with path " + file + "\nError: " + err)
        raise
    return df


def add_columns_to_df(basic_df, columns):
    # Set Multi Index on the dataframe to get the 3d data structure
    try:
        new_df = basic_df.set_index(['Symbol', 'Date'])
    except Exception as err:
        print("index set error")
        print(err)
        raise

    # Add columns to the new df
    for col in columns:
        new_df[col] = -1

    return new_df


def handle_input_arguments():
    """
    Handle input arguments and allow for custom test files
    :return:
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "-test" and len(sys.argv) == 2:
            return "test"
        elif sys.argv[1] == "-f" and len(sys.argv) == 3:
            return "test"
        elif (len(sys.argv) == 2 and sys.argv[1] is not "-test") or (len(sys.argv) == 3 and sys.argv[1] != "-f") or len(sys.argv) > 3:
            print("ERROR: Improper input arguments!\nDefault Test Command:"
                  " \n\tpython main.py -test\nCustom Test File Command:\n\tpython main.py -f <file name>")
            return "error"
        else:
            return "live"


def main():
    file_path = "data/stock_prices_and_returns.csv"
    test_file_path = "data-test/Head_stock_prices_and_returns.csv"
    output_file_path = "data/momentum-features.csv"
    test_output_file = "data-test/test-momentum.csv"
    new_columns = ['RSI', 'Volatility', 'Sharp_Ratio']
    ticker_list = []
    date_list = []

    action = handle_input_arguments()
    if action == "test":
        file_path = test_file_path
        output_file_path = test_output_file
    elif action == "error":
        return 1

    print("input file: " + file_path)
    print("output file: " + output_file_path)

    basic_df = get_dataframe_from_csv(file_path)
    df = add_columns_to_df(basic_df, new_columns)

    # Get Index Lists
    for symbol, mrow in df.groupby(level=0):
        ticker_list.append(symbol)

    for date, mrow in df.groupby(level=1):
        date_list.append(date)

    print('Generating Momentum Features\n-------------------------------------------------------------')
    print('Updating Dataframe with RSI, Volatility, Sharp Ratio and Performance Rank columns......')
    start = time.time()
    for symbol in ticker_list:
        df = momentum.get_stock_rsi_daily(df, symbol)
        df = vol.get_stock_volatility(df, symbol)

    # Get Daily adjusted return rankings based on trailing monthly and yearly prices
    df_yearly, df_monthly = momentum.get_daily_adjusted_stock_return_rankings(df, ticker_list, date_list)

    # Reset multi-index to single index on Symbol
    df_yearly.reset_index(level=1, inplace=True)
    df_monthly.reset_index(level=1, inplace=True)
    print(df_monthly.columns)
    print(df_yearly.columns)

    # Drop duplicate columns to isolate monthly rankings
    try:
        df_monthly.drop(columns=['Open','High', 'Low', 'Close', 'Volume', 'AdjClose', 'Pct_Change_Daily', 'Pct_Change_Monthly',
                                 'Pct_Change_Yearly', 'RSI', 'Volatility'], inplace=True)
    except Exception as err:
        pass

    # Declare Final Dataframe to be stored
    final_df = pd.DataFrame()

    # Loop symbol rows in dataframe and merge to add the monthly return rankings to the yearly
    for symbol in ticker_list:
        tmp = pd.merge(df_yearly.loc[symbol], df_monthly.loc[symbol], on='Date', how='inner')
        df_yearly.loc[symbol, :] = tmp
        tmp['Symbol'] = symbol
        final_df = final_df.append(tmp)

    # Adjusted index before converted or stored
    try:
        final_df.reset_index(level=0, inplace=True)
        final_df.set_index(['Symbol', 'Date'], inplace=True)
        final_df.drop(columns=['Yearly_Return', 'Monthly_Return', 'index'], inplace=True)

    except Exception as err:
        print(err)

    ######################################################
    #  OUTPUT DATA #######################################
    print("Writing to file: " + output_file_path)
    final_df.to_csv(output_file_path, encoding='utf-8', index=True)
    end = time.time()
    print("Process time: " + str(end - start) + " seconds.")
    ######################################################


if __name__ == '__main__':
    main()