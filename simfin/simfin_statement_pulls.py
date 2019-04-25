# -*- coding: utf-8 -*-
"""
Set up interface with SimFin API using quarter-year-statement inputs to reduce 
API hits and avoid duplicitous pulling/storage of data; standardizes creation
of intermediary data sets to load into Postgres DB

TO-DO:
    Set up dynamic dating convention for TTM periods

04/14/2019
Jared Berry
"""
import simfin_setup
import pandas as pd
import requests
import re
import sys

def slugify(value):
    """
    Converts to ASCII. Converts spaces to underscores. Removes characters that
    aren't alphanumerics, underscores, or hyphens. Converts to lowercase.
    Also strips leading and trailing whitespace.
    """
    value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '_', value)

def get_single_statement(sim_ids,
                         api_key,
                         tickers,
                         statement_type,
                         year,
                         quarter
                         ):
    """
    Pulls SimFin financial statement data programatically from the SimFin API. 
    Takes as input a specified year-quarter pair, and statement-type 
    mnemonic. In addition, relies on a list of sim IDs, tickers and user's API 
    key from previously defined functions to interact with the API.
    """

    data = {}
    dfs = []
    for idx, sim_id in enumerate(sim_ids):
        print("Processing {} statements".format(tickers[idx]))
        d = data[tickers[idx]] = {"Line Item": []}
        if sim_id != 0:
            
            if 'TTM' not in quarter:
                period_identifier = quarter + "-" + str(year)
            else: 
                period_identifier = quarter

            if period_identifier not in d:
                d[period_identifier] = []

            url = f'https://simfin.com/api/v1/companies/id/{sim_id}/statements/standardised?stype={statement_type}&fyear={year}&ptype={quarter}&api-key={api_key}'

            content = requests.get(url)
            statement_data = content.json()

            # Collect line item names once; consistent across entities
            if len(d['Line Item']) == 0 and 'values' in statement_data:
                d['Line Item'] = [x['standardisedName'] for x in statement_data['values']]

            if 'values' in statement_data:
                for item in statement_data['values']:
                    d[period_identifier].append(item['valueChosen'])
            else:
                # No data found for time period
                d[period_identifier] = [None for _ in d['Line Item']]

            # Convert to pandas DataFrame and melt down quarter columns
            df = pd.DataFrame(data=d)
            cols = df.columns.drop('Line Item')
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
            melted_df = df.melt(id_vars='Line Item', var_name='date')
    
            # Cast 'Line Item' data across columns
            pivoted_df = pd.pivot_table(melted_df,
                                        values='value',
                                        index=['date'],
                                        columns=['Line Item']
                                        )
            pivoted_df.reset_index(inplace=True)
            pivoted_df['ticker'] = tickers[idx]
    
            # Slugify columns
            pivoted_df.rename(columns=lambda x: slugify(x), inplace=True)
    
            # Append for export
            if not pivoted_df.empty:
                dfs.append(pivoted_df)
        
    return(dfs)

def main(key, statement_type, year, quarter):
    """
    Main execution
    """
    
    # Pull all Sim IDs and tickers from set-up
    api_key = simfin_setup.set_key(key)
    tickers, sim_ids = simfin_setup.load_sim_ids()
    
    # SimFin statements -------------------------------------------------------
    simfin_statements = get_single_statement(sim_ids=sim_ids,
                                         tickers=tickers,
                                         api_key=api_key,
                                         statement_type=statement_type,
                                         year=year,
                                         quarter=quarter)

    # Stack SimFin DataFrames
    simfin_statement_data = pd.concat(simfin_statements, axis=0)

    # Export
    fname = 'simfin_{}_{}_{}'.format(statement_type, quarter, year)
    simfin_statement_data.to_csv('{}.csv'.format(fname), index=False)
    ## dbpath='/Users/syandra/Documents/sqlite-tools-osx-x86-3270200/capstone.db'
    ## con = sqlite3.connect(dbpath)
    ## simfin_statement_data.to_sql(fname, con)

if __name__ == '__main__':
    key = str(sys.argv[1])
    statement_type = str(sys.argv[2])
    year = str(sys.argv[3])
    quarter = str(sys.argv[4])
    main(key, statement_type, year, quarter)