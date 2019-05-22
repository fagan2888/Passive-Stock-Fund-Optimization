def get_dataframe_from_csv(file):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError as err:
        print("FileNotFoundError with path " + file + "\nError: " + err)
        raise
    return df


def add_columns_to_df(basic_df, columns):
    # Set Multi Index on the dataframe to get the 3d data structure
    try:
        new_df = basic_df.set_index(['Symbol', 'Date'])
    except Exception as err:
        print("index set error")
        print(err)
        raise

    # Add columns to the new df
    for col in columns:
        new_df[col] = -1

    return new_df
