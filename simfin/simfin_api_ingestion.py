# -*- coding: utf-8 -*-

import pandas as pd
import requests
import re

def set_key(key):
    """
    Instantiates an API key object to pull from the SimFin API.
    """
    api_key = key
    return(api_key)

def slugify(value):
    """
    Converts to ASCII. Converts spaces to underscores. Removes characters that
    aren't alphanumerics, underscores, or hyphens. Converts to lowercase.
    Also strips leading and trailing whitespace.
    """
    value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '_', value)

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
            sim_ids.append(data[0]['simId'])

    return(sim_ids)

def get_statement_data(sim_ids,
                       api_key,
                       tickers,
                       statement_type,
                       time_periods=["Q1","Q2","Q3","Q4"],
                       year_start=2011,
                       year_end=2019
                       ):
    """
    Pulls SimFin financial statement data programatically from the SimFin API. 
    Takes as input specified time periods, start/end dates, and statement-type 
    mnemonic. In addition, relies on a list of sim IDs, tickers and user's API 
    key from previously defined functions to interact with the API.
    """

    data = {}
    dfs = []
    for idx, sim_id in enumerate(sim_ids):
        print("Processing {} statements".format(tickers[idx]))
        d = data[tickers[idx]] = {"Line Item": []}
        if sim_id is not None:
            for year in range(year_start, year_end + 1):
                for time_period in time_periods:

                    period_identifier = time_period + "-" + str(year)

                    if period_identifier not in d:
                        d[period_identifier] = []

                    url = f'https://simfin.com/api/v1/companies/id/{sim_id}/statements/standardised?stype={statement_type}&fyear={year}&ptype={time_period}&api-key={api_key}'

                    content = requests.get(url)
                    statement_data = content.json()

                    # Collect line item names once; consistent across entities
                    if len(d['Line Item']) == 0:
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
        dfs.append(pivoted_df)
        
    return(data, dfs)
    
def get_single_statement_data(sim_ids,
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
        if sim_id is not None:
            period_identifier = quarter + "-" + str(year)

                if period_identifier not in d:
                    d[period_identifier] = []

                url = f'https://simfin.com/api/v1/companies/id/{sim_id}/statements/standardised?stype={statement_type}&fyear={year}&ptype={quarter}&api-key={api_key}'

                content = requests.get(url)
                statement_data = content.json()

                # Collect line item names once; consistent across entities
                if len(d['Line Item']) == 0:
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
        dfs.append(pivoted_df)
        
    return(data, dfs)
    
def get_shares_data(sim_ids,
                    api_key,
                    tickers
                    ):
    """
    Pulls SimFin data programatically from the SimFin API. Takes as input
    a list of sim IDs, tickers and user's API key from previously defined 
    functions to interact with the API. Pulls all shares data (both figures and
    time periods) for each entity.
    """

    dfs = []
    for idx, sim_id in enumerate(sim_ids):
        print("Processing {} shares data".format(tickers[idx]))
        if sim_id is not None:
            
            url = f'https://simfin.com/api/v1/companies/id/{sim_id}/shares/aggregated?&api-key={api_key}'
            
            # Query SimFin API
            content = requests.get(url)
            shares_data = content.json()
            
            # Convert list of JSON blobs to pandas dataframe
            cols = list(shares_data[0].keys())
            vals = [list(_.values()) for _ in shares_data]
            
            shares_df = pd.DataFrame(vals, columns=cols)
            
            # Light formatting
            shares_df['value'] = pd.to_numeric(shares_df['value'])
            shares_df['ticker'] = tickers[idx]

            # Append for export
            dfs.append(shares_df)
        
    return(dfs)
    
def get_ratios_data(sim_ids,
                    api_key,
                    tickers
                    ):
    """
    Pulls SimFin data programatically from the SimFin API. Takes as input
    a list of sim IDs, tickers and user's API key from previously defined 
    functions to interact with the API. Pulls all shares data (both figures and
    time periods) for each entity.
    """

    dfs = []
    for idx, sim_id in enumerate(sim_ids):
        print("Processing {} shares data".format(tickers[idx]))
        if sim_id is not None:
            
            url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?&api-key={api_key}'
            
            # Query SimFin API
            content = requests.get(url)
            shares_data = content.json()
            
            # Convert list of JSON blobs to pandas dataframe
            cols = list(shares_data[0].keys())
            vals = [list(_.values()) for _ in shares_data]
            
            shares_df = pd.DataFrame(vals, columns=cols)
            
            # Light formatting
            shares_df['value'] = pd.to_numeric(shares_df['value'])
            shares_df['ticker'] = tickers[idx]

            # Append for export
            dfs.append(shares_df)
        
    return(dfs) 

def main(key=key):
    """
    Main execution
    """
    # Set up for pulls
    api_key = set_key(key=key)
    tickers = get_tickers()
    sim_ids = get_sim_ids(tickers, api_key)

    # SimFin statements -------------------------------------------------------
    statements = ['pl', 'bs', 'cf']

    for s in statements:
        simfin_raw, simfin_statements = get_statement_data(sim_ids=sim_ids,
                                                           tickers=tickers,
                                                           api_key=api_key,
                                                           statement_type=s
                                                           )

        # Stack SimFin DataFrames
        simfin_statements_data = pd.concat(simfin_statements, axis=0)

    	# Export
        fname = 'simfin{}'.format(s)
        simfin_statements_data.to_csv('{}.csv'.format(fname))
        ## DBPATH='/Users/syandra/Documents/sqlite-tools-osx-x86-3270200/capstone.db'
        ## con = sqlite3.connect(DBPATH)
        ## simfin_statements_data.to_sql(fname, con)
        
    # SimFin shares -----------------------------------------------------------
    simfin_shares = get_shares_data(sim_ids=sim_ids,
                                    tickers=tickers,
                                    api_key=api_key
                                    )

    # Stack SimFin DataFrames
    simfin_shares_data = pd.concat(simfin_shares, axis=0)

    # Export
    fname = 'simfin_shares'
    simfin_shares_data.to_csv('simfin_shares.csv')
    ## DBPATH='/Users/syandra/Documents/sqlite-tools-osx-x86-3270200/capstone.db'
    ## con = sqlite3.connect(DBPATH)
    ## simfin_shares_data.to_sql(fname, con)    
    
    # SimFin ratios -----------------------------------------------------------
    

if __name__ == '__main__':
    main()
