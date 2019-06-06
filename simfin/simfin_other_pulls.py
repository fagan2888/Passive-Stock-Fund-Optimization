# -*- coding: utf-8 -*-
"""
Set up interface with SimFin API for non-statment pulls to reduce API hits

04/14/2019
Jared Berry
"""

import requests
import pandas as pd
import driver
import simfin_setup


def get_shares_data(sim_ids, api_key, tickers, prices=False):
    """
    Pulls SimFin data programatically from the SimFin API. Takes as input
    a list of sim IDs, tickers and user's API key from previously defined
    functions to interact with the API. Pulls all shares data (both figures and
    time periods) for each entity.
    """

    dfs = []
    for idx, sim_id in enumerate(sim_ids):
        if sim_id != 0:
            print("Processing {} shares data".format(tickers[idx]))

            if prices:

                # Set dynamic API url
                url = f'https://simfin.com/api/v1/companies/id/{sim_id}/shares/prices?&api-key={api_key}'

                # Query SimFin API
                content = requests.get(url)
                shares_data = content.json()

                if "error" in shares_data:
                    continue

                shares_df = pd.DataFrame(shares_data)

            else:

                # Set dynamic API url
                url = f'https://simfin.com/api/v1/companies/id/{sim_id}/shares/aggregated?&api-key={api_key}'

                # Query SimFin API
                content = requests.get(url)
                shares_data = content.json()

                if "error" in shares_data:
                    continue

                # Convert list of JSON blobs to pandas dataframe
                cols = list(shares_data[0].keys())
                vals = [list(_.values()) for _ in shares_data]

                shares_df = pd.DataFrame(vals, columns=cols)

                # Light formatting
                shares_df['value'] = pd.to_numeric(shares_df['value'])

            shares_df['ticker'] = tickers[idx]

            # Append for export
            dfs.append(shares_df)

    return dfs

def get_ratios_data(sim_ids, api_key, tickers):
    """
    Pulls SimFin data programatically from the SimFin API. Takes as input
    a list of sim IDs, tickers and user's API key from previously defined
    functions to interact with the API. Pulls all ratios data (both figures and
    time periods) for each entity.
    """

    dfs = []
    for idx, sim_id in enumerate(sim_ids):
        if sim_id != 0:
            print("Processing {} shares data".format(tickers[idx]))

            url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?&api-key={api_key}'

            # Query SimFin API
            content = requests.get(url)
            ratios_data = content.json()

            # Convert list of JSON blobs to pandas dataframe
            cols = list(ratios_data[0].keys())
            vals = [list(_.values()) for _ in ratios_data]

            ratios_df = pd.DataFrame(vals, columns=cols)

            # Light formatting
            ratios_df['value'] = pd.to_numeric(ratios_df['value'])
            ratios_df['ticker'] = tickers[idx]

            # Append for export
            dfs.append(ratios_df)

    return dfs

def main(key):
    """
    Main execution
    """

    # Pull all Sim IDs and tickers from set-up
    api_key = simfin_setup.set_key(key)
    tickers, sim_ids = simfin_setup.load_sim_ids()

    # SimFin shares -----------------------------------------------------------
    simfin_shares = get_shares_data(sim_ids=sim_ids,
                                    tickers=tickers,
                                    api_key=api_key,
                                    prices=False)

    # Stack SimFin DataFrames
    simfin_shares_data = pd.concat(simfin_shares, axis=0)

    # Export
    fname = 'simfin_shares'
    simfin_shares_data.to_csv('{}.csv'.format(fname), index=False)

    # SimFin share prices -----------------------------------------------------
    simfin_share_prices = get_shares_data(sim_ids=sim_ids,
                                          tickers=tickers,
                                          api_key=api_key,
                                          prices=True)

    # Stack SimFin DataFrames
    simfin_share_price_data = pd.concat(simfin_share_prices, axis=0)

    # Export
    fname = 'simfin_share_prices'
    simfin_share_price_data.to_csv('{}.csv'.format(fname), index=False)

if __name__ == '__main__':
    main(driver.key)
    