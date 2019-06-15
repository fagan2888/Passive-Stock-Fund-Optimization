# -*- coding: utf-8 -*-
"""
Set up interface with SimFin API using quarter-year-statement inputs to reduce
API hits and avoid duplicitous pulling/storage of data; standardizes creation
of intermediary data sets to load into Postgres DB

Credits:
https://medium.com/@SimFin_official/simfin-api-tutorial-6626c6c1dbeb

04/14/2019
Jared Berry
"""

import re
import requests
import pandas as pd
import driver
import simfin_setup

def slugify(value):
    """
    Converts to ASCII. Converts spaces to underscores. Removes characters that
    aren't alphanumerics, underscores, or hyphens. Converts to lowercase.
    Also strips leading and trailing whitespace.
    """
    value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '_', value)

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
        d = data[tickers[idx]] = {"Line Item": []}
        if sim_id != 0:
            print("Processing {} statements".format(tickers[idx]))

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

    return dfs

def main(key, statement_types, years, quarters):
    """
    Main execution
    """

    # Pull all Sim IDs and tickers from set-up
    api_key = simfin_setup.set_key(key)
    tickers, sim_ids = simfin_setup.load_sim_ids()

    # Load process file if boolean specified in driver to avoid extra pulls
    if driver.USE_PROCESS_FILE:
        try:
            process_file = pd.read_csv('process_file.csv')
            cols = process_file.columns.tolist()
        except FileNotFoundError:
            cols = ['statement_type', 'year', 'quarter']
            process_file = pd.DataFrame(columns=cols)

        file_ids = list(zip(process_file['statement_type'].tolist(),
                            process_file['year'].tolist(),
                            process_file['quarter'].tolist()))

    # SimFin statements -------------------------------------------------------
    for s in statement_types:
        for y in years:
            for q in quarters:

                if driver.USE_PROCESS_FILE:
                    file_id = (s, int(y), q)
                    if file_id in file_ids:
                        continue

                simfin_statements = get_single_statement(sim_ids=sim_ids,
                                                         tickers=tickers,
                                                         api_key=api_key,
                                                         statement_type=s,
                                                         year=y,
                                                         quarter=q)

                # Stack SimFin DataFrames
                simfin_statement_data = pd.concat(simfin_statements, axis=0)

                # Export
                fname = 'simfin_{}_{}_{}'.format(s, q, y)
                simfin_statement_data.to_csv('{}.csv'.format(fname), index=False)

                # Record in process file, if boolean specified in driver
                if driver.USE_PROCESS_FILE:
                    new_entry = pd.DataFrame([file_id], columns=cols)
                    process_file = process_file.append(new_entry)

    if driver.USE_PROCESS_FILE:
        process_file.to_csv('process_file.csv', index=False)
    else:
        print("This was run without the process file.")
        print("Be sure to append what you've processed for later ingestion!")

if __name__ == '__main__':
    main(driver.KEY, driver.STATEMENT_TYPES, driver.YEARS, driver.QUARTERS)
    