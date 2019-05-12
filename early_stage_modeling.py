# -*- coding: utf-8 -*-
"""
Early-stage modeling

05/11/2019
Jared Berry
"""

# Import necessary libraries for data preparation/modeling
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load datasets into memory
yahoo = pd.read_csv('stock_price_until_2019_04_28.csv')
simfin = pd.read_csv('simfin\daily_simfin.csv')

# Check dimensions
print("Yahoo! Finance data has {} observations and {} features.".format(yahoo.shape[0], yahoo.shape[1]))
print("Daily SimFin data has {} observations and {} features.".format(simfin.shape[0], simfin.shape[1]))

# Check keys
print("SimFin:")
print(simfin[['ticker', 'date']].head())
print("Yahoo! Finance:")
print(yahoo[['Symbol', 'date_of_transaction']].head())

# Some quick fixes on keys
yahoo['ticker'] = yahoo['Symbol']
yahoo.drop('Symbol', axis=1, inplace=True)

simfin['date_of_transaction'] = simfin['date']
simfin.drop('date', axis=1, inplace=True)

# SET UP TARGET
train = yahoo ## pd.merge(yahoo, simfin, on=['ticker', 'date_of_transaction'])
train.head()

# Testing out creation of a target
train = train.loc[train['ticker'] == 'ABT',:]
train_21 = train['AdjClose'].shift(-21) # WILL NEED TO DO THIS FOR EACH COMPANY
## pd.concat([train, train_21], axis=1)
train_21.name = 'AdjClose_ahead'
y_cont = np.array(100*((train_21 - train['AdjClose'])/train['AdjClose']))
y_disc = np.where(y_cont >= 0, 1, 
         np.where(np.isnan(y_cont), np.nan, 0))

# CHOOSE FEATURES
features = ['High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Year',
            'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear']

X = train[features]
X[['High_diff', 'Low_diff', 'Open_diff', 'Close', 'Volume', 'AdjClose']] =        \
X[['High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose']]  -       \
X[['High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose']].shift(1)
X = X.apply(lambda x: (x - np.mean(x))/np.std(x)).fillna(0) # Impute 0 for now

# Split out mechanically-missing data
test_idx = np.where(np.isnan(y_disc))[0].tolist()

y = np.delete(y_disc, test_idx)
X_holdout = X.loc[X.index[test_idx]]
X = X.drop(X.index[test_idx])

# SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.metrics import confusion_matrix, classification_report

# MODEL
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

tscv = TimeSeriesSplit(n_splits=12).split(X)
gsearch = GridSearchCV(estimator=model, cv=tscv,
                       param_grid=param_search)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
