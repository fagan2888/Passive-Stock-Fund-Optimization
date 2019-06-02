# -*- coding: utf-8 -*-
"""
Pre-modeling set-up

05/18/2019
Jared Berry
"""

# Import necessary libraries for data preparation/EDA
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from collections import defaultdict

# SET-UP ----------------------------------------------------------------------

# Connect to PostgresDB and pull in datasets
engine = create_engine("postgresql://postgres:dfdk#418!@@34.74.173.183/postgres")
                       
# Yahoo! Finance
yahoo=pd.read_sql_query('select * from stock_price', con=engine)
print("Yahoo! Finance features:")
print(yahoo.columns.tolist())

# SimFin Fundamentals
simfindaily=pd.read_sql_query('select * from daily_simfin', con=engine)
print("SimFin features:")
print(simfindaily.columns.tolist())

# Derived momentum features
momentum=pd.read_sql_query('select * from momentum_features', con=engine)
print("Derived features")
print(momentum.columns.tolist())

# S&P 500 index
snp = pd.read_sql_query('select * from spy_stock_price', con = engine)
print("S&P 500")
print(snp.columns.tolist())

# Some quick fixes on keys
simfindaily['date_of_transaction'] = simfindaily['date']
simfindaily.drop('date', axis=1, inplace=True) 

yahoo['ticker'] = yahoo['Symbol']
yahoo.drop('Symbol', axis=1, inplace=True)

momentum['ticker'] = momentum['Symbol']
momentum['date_of_transaction'] = momentum['Date']
momentum.drop(['Symbol', 'Date', 'High', 'Low', 
               'Open', 'Close', 'Volume', 'AdjClose'], 
              axis=1, inplace=True)

snp['snp500_close'] = snp['Adj Close']
snp['snp500_open'] = snp['Open']
snp = snp[['date_of_transaction', 'snp500_close', 'snp500_open']]

# Merge
df = pd.merge(yahoo, momentum, on=['ticker', 'date_of_transaction'])
df = pd.merge(df, simfindaily, how='left', on=['ticker', 'date_of_transaction'])

df = df.sort_values(['ticker','date_of_transaction']).reset_index(drop = True)
df.head()

# Pull out the tickers
tickers = df['ticker'].unique().tolist()

# COMBINED DATA SET FEATURE ENGINEERING ---------------------------------------

# Construct some aggregate financial ratios from the SimFin data
df['eps'] = df['net_income_y'] / df['common_outstanding_basic']
df['pe_ratio'] = df['AdjClose'] / df['eps']
df['debt_ratio'] = df['total_liabilities'] / df['total_equity']
df['debt_to_equity'] = df['total_liabilities'] / df['total_equity']
df['roa'] = df['net_income_y'] / df['total_assets']

# Construct some additional ticker-level returns features
df['open_l1'] = df.groupby('ticker')['Open'].shift(1)
df['open_l5'] = df.groupby('ticker')['Open'].shift(5)
df['open_l10'] = df.groupby('ticker')['Open'].shift(10)

df['return_prev1_open_raw'] = 100*(df['Open'] - df['open_l1'])/df['open_l1']
df['return_prev5_open_raw'] = 100*(df['Open'] - df['open_l5'])/df['open_l5']
df['return_prev10_open_raw'] = 100*(df['Open'] - df['open_l10'])/df['open_l10']

df['close_l1'] = df.groupby('ticker')['AdjClose'].shift(1)
df['close_l5'] = df.groupby('ticker')['AdjClose'].shift(5)
df['close_l10'] = df.groupby('ticker')['AdjClose'].shift(10)

df['return_prev1_close_raw'] = 100*(df['AdjClose'] - df['close_l1'])/df['close_l1']
df['return_prev5_close_raw'] = 100*(df['AdjClose'] - df['close_l5'])/df['close_l5']
df['return_prev10_close_raw'] = 100*(df['AdjClose'] - df['close_l10'])/df['close_l10']

# Compute market betas
betas = np.empty(df.shape[0])
for t in tickers:
    idx = df['ticker'].loc[df['ticker'] == t].index.tolist()
    x_t = df[['date_of_transaction', 'AdjClose']].iloc[idx]

    x_t = pd.merge(x_t, snp, on='date_of_transaction').sort_values('date_of_transaction')

    market_return = np.array(x_t['snp500_close'].tolist())
    asset_return = np.array(x_t['AdjClose'].tolist())

    beta_vector = np.empty(len(asset_return)) * np.nan
    i = 21
    while i < len(beta_vector):
        beta_vector[i] = (np.cov(market_return[:(i-1)], 
                                 asset_return[:(i-1)])[0,1] / 
                          np.var(market_return[:(i-1)]))
        i += 1
        
    betas[idx] = beta_vector
    
df['beta'] = betas

# Features to smooth
to_smooth = ['High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Pct_Change_Daily',
            'Pct_Change_Monthly', 'Pct_Change_Yearly', 'RSI', 'Volatility',
            'Yearly_Return_Rank', 'Monthly_Return_Rank', 'Pct_Change_Class',
            'Rolling_Yearly_Mean_Positive_Days', 'Rolling_Monthly_Mean_Positive_Days', 
            'Rolling_Monthly_Mean_Price', 'Rolling_Yearly_Mean_Price',
            'open_l1', 'open_l5', 'open_l10', 'close_l1', 'close_l5', 'close_l10',
            'return_prev1_open_raw', 'return_prev5_open_raw', 'return_prev10_open_raw',
            'return_prev1_close_raw', 'return_prev5_close_raw', 'return_prev10_close_raw',
            'pe_ratio', 'debt_ratio', 'debt_to_equity', 'roa', 'Momentum_Quality_Monthly', 
            'Momentum_Quality_Yearly', 'SPY_Trailing_Month_Return'
            ]

# Create smoothed variants of specified features
for feature in to_smooth:
    x_to_smooth = np.array(df[feature].tolist())
    col = feature + "_smoothed"
    for t in tickers:
        idx = df['ticker'].loc[df['ticker'] == t].index.tolist()
        x_t = np.array(x_to_smooth[idx].tolist())

        # Compute EMA smoothing of target within ticker
        EMA = 0
        gamma_ = 0.5
        for ti in range(len(x_t)):
            EMA = gamma_*x_t[ti] + (1-gamma_)*EMA
            x_t[ti] = EMA

        x_to_smooth[idx] = x_t
    df[col] = x_to_smooth
    
# Hash the ticker to create a categorical feature
from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features = len(tickers), input_type = 'string')
f = h.transform(df['ticker'])
ticker_features = f.toarray()

# Remove the quarter of pre-SimFin data
train = df[df['date_of_transaction'] >= '2011-03-31'].reset_index(drop=True)

# At the ticker level, lead the AdjClose column by n-trading days
target_gen = train[['ticker', 'date_of_transaction', 'AdjClose', 
                    'Monthly_Return_Rank', 'beta']]
target_gen = pd.merge(target_gen, snp, on='date_of_transaction')

# Loop over specified horizons to generate a number of possible targets
horizons = [1,5,10,21]
rank_threshold = 100
target_dict = defaultdict(list)
for h in horizons:
    n = h # n-day ahead return
    q = h # q-day window
    
    # At the ticker level, lead the AdjClose column n-trading days
    AdjClose_ahead = target_gen.groupby('ticker')['AdjClose'].shift(-n)
    AdjClose_ahead.name = 'AdjClose_ahead'
    
    snp_ahead = target_gen.groupby('ticker')['snp500'].shift(-n)
    snp_ahead.name = 'snp_ahead'
    
    # Raw returns
    target_return = np.array(100*((AdjClose_ahead - target_gen['AdjClose'])/target_gen['AdjClose']))
    
    # Market residualized returns
    target_return_res = target_return - np.array(target_gen['beta'].tolist())*target_return
    
    # Computing all of the returns for the next 21 days (month) relative to today
    aheads = []
    for i in range(0,n+1):
        AdjClose_ahead_i = target_gen.groupby('ticker')['AdjClose'].shift(-i)
        aheads.append(np.array(100*((AdjClose_ahead_i - target_gen['AdjClose'])/target_gen['AdjClose'])))
    
    # Composite, average returns
    target_composite = np.array(pd.DataFrame(aheads).mean(axis=0, skipna=False).tolist())
    
    # q-day moving average of n-day ahead returns, where n=q
    target_gen['returns_ahead'] = 100*((AdjClose_ahead - target_gen['AdjClose'])/target_gen['AdjClose'])
    target_average = np.array(target_gen.groupby('ticker')['returns_ahead'].rolling(q).mean())
    
    # Rank target, binarized
    target_rank = target_gen.groupby('ticker')['Monthly_Return_Rank'].shift(-n)
    target_rank = np.where(np.isnan(target_rank), np.nan,
                  np.where(target_rank < rank_threshold, 1, 0))
    target_rank = target_rank.tolist()
    
    # Simple 'up' target, relative to today
    target_up = np.where(np.isnan(AdjClose_ahead), np.nan,
                np.where(AdjClose_ahead > train['AdjClose'], 1, 0))
    target_up = target_up.tolist()
    
    # Returns, relative to the S&P 500
    snp_return = np.array(100*((AdjClose_ahead - train['AdjClose'])/train['AdjClose']))
    target_rel_return = target_return - snp_return
    
    # 'Up' target, relative to today
    target_rel_up = np.where(np.isnan(AdjClose_ahead) or np.isnan(snp_ahead), np.nan,
                    np.where(AdjClose_ahead > snp_ahead, 1, 0))
    target_rel_up = target_rel_up.tolist()
    
    # Generate keys based on horizon
    return_key = "target_{}_return".format(n)
    return_res_key = "target_{}_return_res".format(n)
    composite_key = "target_{}_composite".format(n)
    average_key = "target_{}_average".format(n)
    rank_key = "target_{}_rank".format(n)
    up_key = "target_{}_up".format(n)
    rel_return_key = "target_{}_rel_return".format(n)
    rel_up_key = "target_{}_rel_up".format(n)
    
    # Store
    target_dict[return_key] = target_return
    target_dict[return_res_key] = target_return_res
    target_dict[composite_key] = target_composite
    target_dict[average_key] = target_average
    target_dict[rank_key] = target_rank
    target_dict[up_key] = target_up
    target_dict[rel_return_key] = target_rel_return
    target_dict[rel_up_key] = target_rel_up
    
# Some diagnostics regarding class imbalance
for key, values in target_dict.items():
    print("% of {} in positive class: {}%".format(key, 100*round(np.nanmean(values),3)))
    
# Add features to dictionary prior to export
target_dict['features'] = train
target_dict['ticker_features'] = ticker_features

# Export
outpath = "model_dictionary.pickle"
with open(outpath, 'wb') as f:
    pickle.dump(target_dict, f)