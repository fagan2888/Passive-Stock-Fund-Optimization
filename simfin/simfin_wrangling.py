# -*- coding: utf-8 -*-
"""
Wrangling code for Simfin statements and shares data sets, preparing for merge
onto daily Yahoo! Finance data

TO-DO:
    Add pulls from instance PostgreSQL DB

04/28/2019
Jared Berry
"""

import simfin_setup
import driver
import pandas as pd
from functools import reduce
import re
import pandas.io.sql as psql
from sqlalchemy import create_engine

def slugify(value):
    """
    Converts to ASCII. Converts spaces to underscores. Removes characters that
    aren't alphanumerics, underscores, or hyphens. Converts to lowercase.
    Also strips leading and trailing whitespace.
    """
    value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '_', value)

def main():
    
    # Read data into memory - Add SQL pulls
    if driver.pull_from_sql:
        print("Not hooked up")
        quit()
    else:
        cf_data = pd.read_csv('qtrly_simfin_cf.csv')
        pl_data = pd.read_csv('qtrly_simfin_pl.csv')
        bs_data = pd.read_csv('qtrly_simfin_bs.csv')
        shares_data = pd.read_csv('simfin_shares.csv')
    
    # Instantiate a quarter, month-day map
    mth_day_map = pd.DataFrame([['Q1', '-03-31'],
                                ['Q2', '-06-30'],
                                ['Q3', '-09-30'],
                                ['Q4', '-12-31']],
                               columns=['period','mth_day'])
    
    # Clean up shares data ----------------------------------------------------
    shares_data = shares_data[(
            (shares_data['measure'] != 'point-in-time') & 
            (shares_data['period'].str.contains('Q') & 
            (shares_data['fyear'] >= 2011))
            )]
    
    shares_data.drop(['type', 'measure', 'date'], axis=1, inplace=True)

    
    shares_data['fyear'] = shares_data['fyear'].map(int).map(str)
    shares_data['qtr_yr'] = shares_data[['period', 'fyear']].apply(lambda x: '-'.join(x), axis=1)
    
    shares_data = pd.merge(shares_data, mth_day_map, on='period')
    shares_data['date'] = shares_data[['fyear', 'mth_day']].apply(lambda x: ''.join(x), axis=1)

    shares_data.drop(['fyear', 'period', 'mth_day'], axis=1, inplace=True)
    
    # Cast 'Line Item' data across columns
    shares_clean = pd.pivot_table(shares_data,
                                values='value',
                                index=['date', 'qtr_yr', 'ticker'],
                                columns=['figure']
                                )
    shares_clean.reset_index(inplace=True)

    # Slugify columns
    shares_clean.rename(columns=lambda x: slugify(x), inplace=True)
    
    # Create a more granular quarter-year-date map
    qtr_yr_map = shares_clean[['date', 'qtr_yr']].drop_duplicates().reset_index().drop('index', axis=1)
    
    # Clean up statements data ------------------------------------------------
    statements_all = [cf_data, pl_data, bs_data]
    statements_clean = []
    
    for s_df in statements_all:
    
        # Identify duplicate ticker-quarter pairs
        tq_dups = s_df.groupby(['date', 'ticker']).size().reset_index()
        tq_dups = tq_dups[tq_dups[0] > 1].drop(0, axis=1)
        
        # Filter down to these entities
        s_dups = pd.merge(s_df.reset_index(), tq_dups, how='inner', on=['date', 'ticker'])
        
        # Create counts of populated entries, by row
        s_dups['n_pop'] = s_dups.apply(lambda x: x.count(), axis=1)
        
        # Identify row indexes of most populated duplicated ticker-quarter pairs
        dups_to_drop = s_dups[['date','ticker','index','n_pop']].sort_values('n_pop', ascending=False).groupby(['date','ticker']).head(1)
        dups_to_drop_idx = dups_to_drop['index'].tolist()
        
        # Drop duplicates
        s_df_clean = s_df.drop(dups_to_drop_idx, axis=0)
        
        # Prepare date/ticker columns for merge
        s_df_clean['qtr_yr'] = s_df_clean['date']
        s_df_clean.drop('date', axis=1, inplace=True)
        s_df_clean['ticker'] = s_df_clean['ticker'].str.upper()
        
        # Merge in dates
        s_df_clean = pd.merge(s_df_clean, qtr_yr_map, how='inner', on='qtr_yr')
        
        # Reorder columns to keep keys at the front
        cols = s_df_clean.columns.tolist()
        
        key_cols = ['ticker', 'qtr_yr', 'date']
        for k in key_cols:
            cols.remove(k)
            
        cols = key_cols + cols
        
        s_df_clean = s_df_clean[cols]
        
        statements_clean.append(s_df_clean)
        
    # Create a mapping of all tickers to all quarters for more stable merges
    tickers = simfin_setup.get_tickers()
    quarters = qtr_yr_map['date'].tolist()
    tic_qtr_map = [[t, q] for t in tickers for q in quarters]
    tic_qtr_map = pd.DataFrame(tic_qtr_map, columns=['ticker', 'date'])
    tic_qtr_map = pd.merge(tic_qtr_map, qtr_yr_map, how='inner', on='date')

    # Merge all data sets together
    dfs_to_reduce = [tic_qtr_map] + statements_clean + [shares_clean]
    qtrly_simfin = reduce(lambda left, right: pd.merge(left, right, how='left', on=key_cols), dfs_to_reduce)
    
    # Remove rows where all fields are missing
    qtrly_simfin['n_pop'] = qtrly_simfin.apply(lambda x: x.count(), axis=1)
    qtrly_simfin = qtrly_simfin[qtrly_simfin['n_pop'] > len(key_cols)].reset_index().drop(['index','n_pop'], axis=1)
    
    # Remove columns which are missing for more than a specified threshold
    n_obs = qtrly_simfin.shape[0]
    pct_msgs = qtrly_simfin.isnull().sum() / n_obs
    cols_to_keep = pct_msgs[pct_msgs < 0.15].index.tolist()
    
    qtrly_simfin = qtrly_simfin[cols_to_keep]
    
    # Generate a daily frequency dataframe
    daily_dates = pd.date_range(start=qtr_yr_map['date'].min(), 
                                end=qtr_yr_map['date'].max())
    daily_dates = [str(x)[:10] for x in daily_dates]
    daily_df_map = pd.DataFrame(daily_dates, columns=['date'])
    
    # Forward fill at the ticker-quarter level
    pop_tickers = qtrly_simfin['ticker'].drop_duplicates().tolist()
    
    daily_dfs = []
    for t in pop_tickers:
        daily_df = pd.merge(daily_df_map,
                            qtrly_simfin[qtrly_simfin['ticker'] == t],
                            how='left',
                            on='date').ffill()
        
        daily_dfs.append(daily_df)
    
    # Reduce by row concatenation
    daily_simfin = pd.concat(daily_dfs).reset_index().drop('index', axis=1)
    
    # Create date key for Yahoo! Finance merge
    daily_simfin['date_of_transaction'] = daily_simfin['date'].str.slice(5,7) + '/' + \
                                          daily_simfin['date'].str.slice(8,10) + '/' + \
                                          daily_simfin['date'].str.slice(0,4)
    daily_simfin['date_of_transaction'] = daily_simfin['date_of_transaction'].str.replace('^0', '')
    daily_simfin['date_of_transaction'] = daily_simfin['date_of_transaction'].str.replace('/0', '/')
                                          
    # Export
    daily_simfin.to_csv('daily_simfin.csv', index=False)
    ## daily_simfin.to_sql('daily_simfin', con=engine, if_exists='replace')
    
if __name__ == '__main__':
    main()    
    