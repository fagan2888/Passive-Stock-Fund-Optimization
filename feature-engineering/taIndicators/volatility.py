import pandas as pd
from talib import RSI
import numpy as np


ROLLING_WINDOW = 21


def get_time_series_adjusted_close(time_series_df, ticker):
    return time_series_df.loc[ticker].loc[::, 'AdjClose'].values


def get_stock_volatility(time_series_df, ticker):
    """
    Compute volatility and sharp ratio with an intermediate (monthly)
    term rolling window
    """
    close = get_time_series_adjusted_close(time_series_df, ticker)
    close_series = pd.Series(close)
    roller = close_series.rolling(ROLLING_WINDOW)
    vol_vec = roller.std(ddof=0)
    tmp_vol_df = pd.DataFrame(data=vol_vec, columns=['Volatility'])
    time_series_df.loc[ticker, 'Volatility'] = (tmp_vol_df['Volatility'].values / close) * 12
    time_series_df.loc[ticker, 'Sharp_Ratio'] = time_series_df.loc[ticker, "Pct_Change_Monthly"].values / ((tmp_vol_df['Volatility'].values / close) * 12)
    return time_series_df