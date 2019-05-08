# -*- coding: utf-8 -*-
"""
Code to stack SimFin intermediaries/load into Postgres DB

05/08/2019
Jared Berry
"""

import simfin_setup
import pandas as pd
import pandas.io.sql as psql
from sqlalchemy import create_engine

# Set up connection to Postgres DB
## engine = create_engine(r'postgresql://some:user@host/db')
## c = engine.connect()
## conn = c.connection

# Use process file to create all relevant file extensions
process_file = pd.read_csv('process_file.csv')
process_file = process_file.sort_values(['statement_type', 'year', 'quarter'])
process_file['year'] = process_file['year'].map(str)
exts = process_file[['statement_type', 'quarter', 'year']].apply(lambda x: '_'.join(x), axis=1).tolist()

# Loop through file extensions, appending to SQL database/running list of csvs
pls = []; bss = []; cfs = []

for e in exts:
    file = 'simfin_{}.csv'.format(e)
    statement = pd.read_csv(file)
    
    if 'pl' in e:
        pls.append(statement)
        ## statement.to_sql('qtrly_simfin_pl', con=engine, if_exists='append')
    if 'cf' in e:
        cfs.append(statement)
        ## statement.to_sql('qtrly_simfin_cf', con=engine, if_exists='append')
    if 'bs' in e:
        bss.append(statement)
        ## statement.to_sql('qtrly_simfin_bs', con=engine, if_exists='append')

# Load shares data into Postgres DB
shares_data = pd.read_csv('simfin_shares.csv')
## shares_data.to_sql('qtrly_simfin_shares', con=engine, if_exists='replace')

# Export csvs to pull in wrangling given driver parameters
pls_df = pd.concat(pls, axis=0)
pls_df.to_csv('qtrly_simfin_pl.csv', index=False)

cfs_df = pd.concat(cfs, axis=0)
cfs_df.to_csv('qtrly_simfin_cf.csv', index=False)

bss_df = pd.concat(bss, axis=0)
bss_df.to_csv('qtrly_simfin_bs.csv', index=False)