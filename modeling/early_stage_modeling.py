# -*- coding: utf-8 -*-
"""
Early-stage modeling

05/18/2019
Jared Berry
"""

# SET-UP ----------------------------------------------------------------------

# Import necessary libraries for data preparation/modeling
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load datasets into memory
yahoo = pd.read_csv('stock_price_until_2019_04_28.csv')
drvd = pd.read_csv('moving-avg-momentum.csv')
simfin = pd.read_csv('simfin\daily_simfin.csv')

# Check dimensions
print("Yahoo! Finance data has {} observations and {} features.".format(yahoo.shape[0], yahoo.shape[1]))
print("Derived data has {} observations and {} features.".format(drvd.shape[0], drvd.shape[1]))
print("Daily SimFin data has {} observations and {} features.".format(simfin.shape[0], simfin.shape[1]))

# Check keys
print("Yahoo! Finance:")
print(yahoo[['Symbol', 'date_of_transaction']].head())
print("Derived (Yahoo! Finance)")
print(drvd[['Symbol', 'Date']].head())
print("SimFin:")
print(simfin[['ticker', 'date']].head())

# Some quick fixes on keys
yahoo['ticker'] = yahoo['Symbol']
yahoo.drop('Symbol', axis=1, inplace=True)

drvd['ticker'] = drvd['Symbol']
drvd['date_of_transaction'] = drvd['Date']
drvd.drop(['Symbol', 'Date', 'High', 'Low', 
           'Open', 'Close', 'Volume', 'AdjClose'], 
          axis=1, inplace=True)

simfin['date_of_transaction'] = simfin['date']
simfin.drop('date', axis=1, inplace=True)

# MERGE -----------------------------------------------------------------------

train = pd.merge(yahoo, drvd, on=['ticker', 'date_of_transaction'])
train = pd.merge(train, simfin, how='left', on=['ticker', 'date_of_transaction'])

train = train.sort_values(['ticker','date_of_transaction'])
train = train[train['date_of_transaction'] >= '2011-03-31'].reset_index().drop('index', axis=1)
train.head()

# FEATURE ENGINEERING (ADDITIONAL) --------------------------------------------

train['eps'] = train['net_income_y'] / train['common_outstanding_basic']
train['pe_ratio'] = train['AdjClose'] / train['eps']
train['debt_ratio'] = train['total_liabilities'] / train['total_equity']
train['debt_to_equity'] = train['total_liabilities'] / train['total_equity']
train['roa'] = train['net_income_y'] / train['total_assets']
 
train = train.loc[train['ticker'] == 'A',:]

# SET UP TARGET ---------------------------------------------------------------

# Specify ranges
n = 1 # n-day ahead return
q = 21 # q-day window

# At the ticker level, lead the AdjClose column by n-trading days
target_gen = train[['ticker', 'date_of_transaction', 'AdjClose']]
AdjClose_ahead = target_gen.groupby('ticker')['AdjClose'].shift(-n)
AdjClose_ahead.name = 'AdjClose_ahead'

target_raw = np.array(100*((AdjClose_ahead - train['AdjClose'])/train['AdjClose']))

# Computing all of the returns for the next 21 days (month) relative to today
aheads = []
for i in range(0,22):
    aheads.append(np.array(100*((train['AdjClose'].shift(-i) - train['AdjClose'])/train['AdjClose'])))

# Average of those returns
target = np.array(pd.DataFrame(aheads).mean(axis=0, skipna=False).tolist())
target_raw = np.array(100*((train['AdjClose'].shift(-1) - train['AdjClose'])/train['AdjClose']))

# CHOOSE FEATURES -------------------------------------------------------------
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Year',
            'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Pct_Change_Daily',
            'Pct_Change_Monthly', 'Pct_Change_Yearly', 'RSI', 'Volatility',
            'Yearly_Return_Rank', 'Monthly_Return_Rank', 'Pct_Change_Class',
            'Rolling_Yearly_Mean_Positive_Days', 'Rolling_Monthly_Mean_Positive_Days', 
            'Rolling_Monthly_Mean_Price', 'Rolling_Yearly_Mean_Price',
            'pe_ratio', 'debt_ratio', 'debt_to_equity', 'roa'
            #'total_assets', 'total_equity', 'total_liabilities', 'net_income_y',
            #'operating_expenses', 'gross_profit', 'retained_earnings', 'revenue'
            ]

X = train[features]
X = X.apply(lambda x: (x - np.mean(x))/np.std(x)).fillna(0)

# PARTITIONING ----------------------------------------------------------------

# Hold out mechanically-missing data
test_idx = np.where(np.isnan(target_raw))[0].tolist()

y = np.delete(target, test_idx)
X_holdout = X.loc[X.index[test_idx]]
X = X.drop(X.index[test_idx])
   
# LightGBM --------------------------------------------------------------------

from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# Convert to NumPy arrays - store feature names
feature_names = X.columns.tolist()
features = np.array(X)
labels = y.copy()

EMA = 0
gamma = 0.25
for ti in range(len(labels)):
    EMA = gamma*labels[ti] + (1-gamma)*EMA
    labels[ti] = EMA   

pd.DataFrame(list(zip(y.copy(), labels))).plot()

# Generate splits
splits = TimeSeriesSplit(n_splits=12)

# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))
    
# Empty array for out of fold validation predictions
out_of_fold = np.zeros(features.shape[0])
    
# Lists for recording validation and training scores
valid_scores = []
train_scores = []

# Iterate through each fold
for train_indices, valid_indices in splits.split(X): 
        
    # Training data for the fold
    train_features, train_labels = features[train_indices], labels[train_indices]
    #Validation data for the fold
    valid_features, valid_labels = features[valid_indices], labels[valid_indices]
    
    EMA = 0
    gamma = 1
    for ti in range(len(train_labels)):
        EMA = gamma*train_labels[ti] + (1-gamma)*EMA
        train_labels[ti] = EMA 
        
    train_labels = np.where(train_labels >= 0, 1, 0)
    valid_labels = np.where(valid_labels >= 0, 1, 0)
    
    # Create the bst
    bst = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                            class_weight = 'balanced', learning_rate = 0.01,
                            ## max_bin = 250, num_leaves = 1000,
                            reg_alpha = 0.1, reg_lambda = 0.1, 
                            subsample = 0.8, random_state = 101)
        
    # Train the bst
    bst.fit(train_features, train_labels, eval_metric = 'auc',
            eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
            eval_names = ['valid', 'train'],
            early_stopping_rounds = 100, verbose = 200)
    
    # Record the best iteration
    best_iteration = bst.best_iteration_
        
    # Record the feature importances
    feature_importance_values += bst.feature_importances_ / splits.n_splits

    # Record the out of fold predictions
    out_of_fold[valid_indices] = bst.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
    # Record the best score
    print(bst.best_score_['valid'])
    valid_score = bst.best_score_['valid']['auc']
    train_score = bst.best_score_['train']['auc']
        
    valid_scores.append(valid_score)
    train_scores.append(train_score) 
  
print("\n")
print(valid_scores)
print("\n")
print(np.mean(valid_scores))
print("\n")
print(sorted(list(zip(feature_names, feature_importance_values)), key=lambda x: x[1], reverse=True))

# PanelSplits -----------------------------------------------------------------

# Create copies for panel-level regressions
X_p = X.copy(deep=True)
y_p = target_raw.copy()

# Indexes of hold-out test data (the 21 days of data preceding the present day)
test_idx = np.where(np.isnan(y_p))[0].tolist()

# In order to ensure grouping is done properly, remove this data from a ticker-identification set as well
tickers = train['ticker'].unique().tolist()
ticker_locs = train['ticker'].drop(train.index[test_idx]).reset_index().drop('index', axis=1)

def PanelSplit(n_folds, groups, window=False):
    
    # Generate a list of the indexes of the indexes for each entity in the data
    idx_list = [ticker_locs.loc[ticker_locs['ticker'] == t].index.tolist() for t in tickers]
    
    # Storage for indexes 
    train_splits = [[] for _ in range(n_folds)]
    valid_splits = [[] for _ in range(n_folds)]
    
    for idx in idx_list:
        splits = TimeSeriesSplit(n_splits=n_folds)
        fold = 0
        for train_indices, valid_indices in splits.split(idx):
            train_splits[fold] = train_splits[fold] + np.array(idx)[train_indices].tolist()
            valid_splits[fold] = valid_splits[fold] + np.array(idx)[valid_indices].tolist()
            fold += 1
            
    panel_splits = list(zip(train_splits, valid_splits))
    
    return(panel_splits)
    
def PanelSplit(n_folds, groups):    
    
    by_ticker_index = (groups.groupby('ticker')
                       .apply(lambda x: x.reset_index(drop=True))
                       .drop('ticker', axis=1)
                       .reset_index()
                       .rename({'level_1':'tsidx'}, axis=1)
                       )
    
    ticker_range = by_ticker_index['tsidx'].unique().tolist()
    ticker_range = sorted(ticker_range)
    
    splits = TimeSeriesSplit(n_splits=n_folds)
    
    for train_indices, valid_indices in splits.split(ticker_range):
        panel_train_indices = by_ticker_index[by_ticker_index['tsidx'].isin(train_indices)].index.tolist()
        panel_valid_indices = by_ticker_index[by_ticker_index['tsidx'].isin(valid_indices)].index.tolist()
        yield panel_train_indices, panel_valid_indices
        
def WindowSplit(window, groups, panel):    
    
    wparams = window.split(':')
    wtrain = int(wparams[0])
    wvalid = int(wparams[1])
    witer = int(wparams[2])
    
    by_ticker_index = (groups.groupby('ticker')
                       .apply(lambda x: x.reset_index(drop=True))
                       .drop('ticker', axis=1)
                       .reset_index()
                       .rename({'level_1':'tsidx'}, axis=1)
                       )
    
    ticker_range = by_ticker_index['tsidx'].unique().tolist()
    ticker_range = sorted(ticker_range)
    
    stop = 0
    start = (max(ticker_range) % witer) + 1
    if panel:
        while stop < max(ticker_range):
            train_indices = np.arange(start, start + wtrain).tolist()
            valid_indices = np.arange(start + wtrain, start + wtrain + wvalid).tolist()
            stop = max(valid_indices)
            start += witer
            yield train_indices, valid_indices
    else:
        while stop < max(ticker_range):
            train_indices = np.arange(start, start + wtrain).tolist()
            valid_indices = np.arange(start + wtrain, start + wtrain + wvalid).tolist()
            stop = max(valid_indices)
            start += witer
            panel_train_indices = by_ticker_index[by_ticker_index['tsidx'].isin(train_indices)].index.tolist()
            panel_valid_indices = by_ticker_index[by_ticker_index['tsidx'].isin(valid_indices)].index.tolist()
            yield panel_train_indices, panel_valid_indices      

panel_splitter = PanelSplit(6, ticker_locs)
for x, y in panel_splitter:
    print(x[:100])
    print(y[:50])