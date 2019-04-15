# -*- coding: utf-8 -*-
"""
Set up SimFin IDs to interface with SimFin API and reduce duplicitous API hits

04/14/2019
Jared Berry
"""

import pandas as pd
import requests
import sys

def set_key(key):
    """
    Instantiates an API key object to pull from the SimFin API.
    """
    api_key = key
    return(api_key)
    
def get_tickers():
    """
    Gets S&P 500 tickers from Wikipedia.
    """
    spylist = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            )
    tickers = spylist[0]["Symbol"].tolist()
    return(tickers)

def get_sim_ids(tickers, api_key):
    """
    Pulls SimFin IDs to pass into programatic API pulls. Takes as input a list
    of standardized tickers, and a SimFin API key.
    """
    sim_ids = []
    for ticker in tickers:

        url = f'https://simfin.com/api/v1/info/find-id/ticker/{ticker}?api-key={api_key}'
        content = requests.get(url)
        data = content.json()
        print(data)

        if "error" in data or len(data) < 1:
            sim_ids.append(None)
        else:
            for i in range(len(data)):
                sim_ids.append((data[i]['ticker'], data[i]['simId']))

    return(sim_ids)
    
def load_sim_ids():
    """
    Loads SimFin IDs and tickers generated in the simfin_setup.py execution
    TO-DO: Pull from instance Postgres DB once loaded, instead of local csv
    """
    ticker_id_map = pd.read_csv('ticker_id_map.csv')
    tickers = ticker_id_map['ticker'].tolist()
    sim_ids = ticker_id_map['sim_id'].fillna(0).astype('int').tolist()
    return(tickers, sim_ids)
    
def main(key):
    """
    Main execution
    """
    
    # Pull all Sim IDs associated with S&P 500 tickers
    api_key = set_key(key)
    tickers = get_tickers()
    sim_ids = get_sim_ids(tickers, api_key)
    
    # Create dataframe to export or load directly into PostgreSQL database
    ticker_id_map = pd.DataFrame(sim_ids, columns=["ticker", "sim_id"])
    
    # Export
    fname = 'ticker_id_map'
    ticker_id_map.to_csv('{}.csv'.format(fname), index=False)
    ## dbpath ='/Users/syandra/Documents/sqlite-tools-osx-x86-3270200/capstone.db'
    ## con = sqlite3.connect(dbpath)
    ## ticker_id_map.to_sql(fname, con)   
    
if __name__ == '__main__':
    key = str(sys.argv[1])
    main(key)