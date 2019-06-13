
START_DATE = '2011-01-03'
END_DATE = '2019-04-03'
RANK_RECALCULATE = 1
YEARLY_TRADING_DAYS = 252
MONTHLY_TRADING_DAYS = 21


def get_time_series_adjusted_close(time_series_df, ticker):
    return time_series_df.loc[ticker].loc[::, 'AdjClose'].values
