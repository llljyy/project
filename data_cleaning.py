def drop_missing_values(df, threshold=0.5, axis=0, subset=None):
    """
    Drops missing values from a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to drop missing values from.
    threshold : float, optional (default=0.5)
        The threshold for the proportion of missing values in a row/column to be dropped.
        If a row/column has missing values in more than `threshold` proportion, it will be dropped.
    axis : int, optional (default=0)
        The axis along which to drop missing values. 0 for rows, 1 for columns.
    subset : list or tuple, optional (default=None)
        The list of column names to consider for dropping missing values.
        If None, all columns are considered.

    Returns:
    --------
    pandas.DataFrame
        The dataframe with missing values dropped.
    """
    if subset is not None:
        df = df[subset]
    if axis == 0:
        # Drop rows with missing values
        return df.dropna(thresh=int(threshold * len(df.columns)))
    elif axis == 1:
        # Drop columns with missing values
        return df.dropna(axis=1, thresh=int(threshold * len(df.index)))
    else:
        raise ValueError("Axis must be 0 or 1.")


def fill_missing_values(df, fill_value=None, method='mean', subset=None):
    """
    Fills missing values in a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to fill missing values in.
    fill_value : scalar or dict, optional (default=None)
        The value or dictionary of values to use for filling missing values.
        If None, `method` is used to fill missing values.
    method : str, optional (default='mean')
        The method used to fill missing values. Supported methods are 'mean', 'median', 'mode',
        and 'ffill' (forward fill). Ignored if `fill_value` is not None. Default is 'mean'.
    subset : list or tuple, optional (default=None)
        The list of column names to consider for filling missing values.
        If None, all columns are considered.

    Returns:
    --------
    pandas.DataFrame
        The dataframe with missing values filled.
    """
    if subset is not None:
        df = df[subset]
    if fill_value is not None:
        return df.fillna(fill_value)
    elif method == 'mean':
        return df.fillna(df.mean())
    elif method == 'median':
        return df.fillna(df.median())
    elif method == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif method == 'ffill':
        return df.ffill()
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods are 'mean', 'median', 'mode', and 'ffill'.")
    

def remove_outliers(df, threshold=3):
    """
    Remove outliers from a Pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to remove outliers from.
    threshold : int or float, optional (default=3)
        The threshold for identifying outliers using the z-score.
        Observations with a z-score greater than or equal to
        `threshold` will be considered outliers.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with outliers removed.
    """
    
    from scipy.stats import zscore

    # Calculate the z-score for each observation in the DataFrame
    zscores = df.apply(zscore)

    # Identify outliers based on the z-score
    outliers = (zscores.abs() >= threshold).any(axis=1)

    # Remove outliers from the DataFrame
    df_clean = df[~outliers]

    return df_clean


def group_and_aggregate(df, group_cols, agg_dict):
    """
    Group and aggregate data in a dataframe.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        group_cols (list): A list of column names to group by.
        agg_dict (dict): A dictionary of columns to aggregate and their respective aggregate functions.
    
    Returns:
        pandas.DataFrame: The grouped and aggregated dataframe.
    """
    # Group by the specified columns and apply the specified aggregate functions
    grouped_df = df.groupby(group_cols).agg(agg_dict)
    
    # Flatten the column index of the resulting dataframe
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
    
    # Reset the index to turn the group columns back into regular columns
    grouped_df.reset_index(inplace=True)
    
    return grouped_df


def solve_data_entry_errors(df, column, expected_values = None):
    """
    Checks if a column in a DataFrame contains unexpected values and replaces them with NaNs.

    Parameters:
        df (pandas.DataFrame): The DataFrame to check for data entry errors.
        column (str): The name of the column to check for data entry errors.
        expected_values (list): A list of expected values for the column.

    Returns:
        pandas.DataFrame: The DataFrame with any unexpected values replaced with NaNs.
    """
    import pandas as pd

    # Create a copy of the DataFrame to avoid modifying the original
    df_clean = df.copy()

    # Convert the column to lowercase and remove leading/trailing whitespace
    df_clean[column] = df_clean[column].str.lower().str.strip()

    if expected_values is not None:
        # Replace any unexpected values with NaNs
        mask = ~df_clean[column].isin(expected_values)
        df_clean.loc[mask, column] = pd.NA

    return df_clean


def pivot_data(df, index_col, columns_col, values_col):
    """
    Pivot and reshape data in a pandas DataFrame based on specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame to pivot.
        index_col (str or list of str): The column(s) to use as the index in the pivoted table.
        columns_col (str): The column to use as the column headers in the pivoted table.
        values_col (str): The column to use as the values in the pivoted table.

    Returns:
        pandas.DataFrame: The pivoted and reshaped DataFrame.
    """
    import pandas as pd
    # Pivot the data based on the specified columns
    pivoted_df = pd.pivot_table(df, index=index_col, columns=columns_col, values=values_col)

    # Reset the index so the pivot table is a DataFrame
    pivoted_df = pivoted_df.reset_index()

    return pivoted_df
