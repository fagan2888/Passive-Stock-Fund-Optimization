# -*- coding: utf-8 -*-
"""
Code to stack SimFin intermediaries/load into Postgres DB

05/08/2019
Jared Berry
"""

import pandas as pd

# Set up connection to Postgres DB
# from sqlalchemy import create_engine
# engine = create_engine("postgresql://postgres:dfdk#418!@@34.74.173.183/postgres")
# c = engine.connect()
# conn = c.connection

# Use process file to create all relevant file extensions
PROCESS_FILE = pd.read_csv('PROCESS_FILE.csv')
PROCESS_FILE = PROCESS_FILE.sort_values(['statement_type', 'year', 'quarter'])
PROCESS_FILE['year'] = PROCESS_FILE['year'].map(str)
exts = (PROCESS_FILE[['statement_type', 'quarter', 'year']]
        .apply(lambda x: '_'.join(x), axis=1)
        .tolist())

# Loop through file extensions, appending to SQL database/running list of csvs
pls = []
bss = []
cfs = []

for e in exts:
    file = 'simfin_{}.csv'.format(e)
    statement = pd.read_csv(file)

    if 'pl' in e:
        pls.append(statement)
        # statement.to_sql('qtrly_simfin_pl', con=engine, if_exists='append')
    if 'cf' in e:
        cfs.append(statement)
        # statement.to_sql('qtrly_simfin_cf', con=engine, if_exists='append')
    if 'bs' in e:
        bss.append(statement)
        # statement.to_sql('qtrly_simfin_bs', con=engine, if_exists='append')

# Load shares data into Postgres DB
shares_data = pd.read_csv('simfin_shares.csv')
# shares_data.to_sql('qtrly_simfin_shares', con=engine, if_exists='replace')

# Export csvs to pull in wrangling given driver parameters
pd.concat(pls, axis=0).to_csv('qtrly_simfin_pl.csv', index=False)
pd.concat(cfs, axis=0).to_csv('qtrly_simfin_cf.csv', index=False)
pd.concat(bss, axis=0).to_csv('qtrly_simfin_bs.csv', index=False)
