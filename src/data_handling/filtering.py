from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from datetime import datetime


def chunk_data(df, num_chunks=os.cpu_count()):
    """
    Split the DataFrame into a specified number of chunks.

    Args:
        df (pandas.DataFrame): The DataFrame to be split into chunks.
        num_chunks (int): Number of chunks (default is the number of CPU cores).

    Returns:
        list of pandas.DataFrame: A list of DataFrames, each representing a chunk of the original DataFrame.

    Example:
    >>> data = {'date': ["2024-06-23", "2024-07-01", "2024-06-01", "2024-06-30", "2023-12-31"],
    ...         'value': [5, 15, 10, 5, 4]}
    >>> df = pd.DataFrame(data)
    >>> chunks = chunk_data(df, 2)
    >>> len(chunks)
    2
    >>> sum(len(chunk) for chunk in chunks) == len(df)
    True
    """
    num_rows = len(df)
    chunk_size = num_rows // num_chunks
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
    chunks.append(df.iloc[(num_chunks-1)*chunk_size:])
    chunks = [chunk.reset_index(drop=True) for chunk in chunks]

    return chunks


def filter_data_by_value(val, val_range):
    """
    Check if a value falls within a specified range.

    Args:
        x (int): The value to check.
        val_range (tuple of int): A tuple containing two integers, representing the inclusive range.

    Returns:
        bool: True if x is within the range [val_range[0], val_range[1]], otherwise False.

    Examples:
    >>> filter_data_by_value(5, (1, 10))
    True
    >>> filter_data_by_value(15, (1, 10))
    False
    >>> filter_data_by_value(10, (10, 20))
    True
    >>> filter_data_by_value(5, (5, 5))
    True
    >>> filter_data_by_value(4, (5, 5))
    False
    """
    if val_range[0] <= val <= val_range[1]:
        return True
    else:
        return False


def filter_data_by_date(date_val, date_range):
    """
    Check if a date falls within a specified date range.

    Args:
        date_val (str): The date to check, in the format "YYYY-MM-DD".
        date_range (tuple of str): A tuple containing two strings, representing the inclusive date range in the format "YYYY-MM-DD".

    Returns:
        bool: True if date_val is within the range [date_range[0], date_range[1]], otherwise False.

    Examples:
    >>> filter_data_by_date("2024-06-23", ("2024-06-01", "2024-06-30"))
    True
    >>> filter_data_by_date("2024-07-01", ("2024-06-01", "2024-06-30"))
    False
    >>> filter_data_by_date("2024-06-01", ("2024-06-01", "2024-06-30"))
    True
    >>> filter_data_by_date("2024-06-30", ("2024-06-01", "2024-06-30"))
    True
    >>> filter_data_by_date("2023-12-31", ("2024-01-01", "2024-12-31"))
    False
    """
    date_range = (datetime.strptime(date_range[0], '%Y-%m-%d'), datetime.strptime(date_range[1], '%Y-%m-%d'))
    date_val = datetime.strptime(date_val, '%Y-%m-%d')

    if date_range[0] <= date_val <= date_range[1]:
        return True
    else:
        return False


def filter_data_serially(func, *args):
    """
    Filter rows of a DataFrame based on a specified function applied serially.

    Args:
        func (function): The function to apply to each row of the DataFrame. It should accept arguments that include values from the DataFrame.
        *args: The arguments to pass to the function, which should include the DataFrame to filter and the additional parameters required by `func`.

    Returns:
        pandas.DataFrame: A DataFrame containing only the rows where the function returns True.

    Examples:
    >>> data = {'date': ["2024-06-23", "2024-07-01", "2024-06-01", "2024-06-30", "2023-12-31"],
    ...         'value': [5, 15, 10, 5, 4]}
    >>> df = pd.DataFrame(data)
    >>> date_range = ("2024-06-01", "2024-06-30")
    >>> filter_data_serially(filter_data_by_date, df, date_range)
             date  value
    0  2024-06-23      5
    1  2024-06-01     10
    2  2024-06-30      5

    >>> value_range = (1, 10)
    >>> filter_data_serially(filter_data_by_value, df, value_range)
             date  value
    0  2024-06-23      5
    1  2024-06-01     10
    2  2024-06-30      5
    3  2023-12-31      4
    """
    df = args[0]
    idx_filtered = []

    if func == filter_data_by_value:
        for idx, row in df.iterrows():
            if func(row['value'], args[1]):
                idx_filtered.append(idx)
    elif func == filter_data_by_date:
        for idx, row in df.iterrows():
            if func(row['date'], args[1]):
                idx_filtered.append(idx)

    filtered_df = df.iloc[idx_filtered]
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def filter_data_parallely(func, *args):
    """
    Filter rows of a DataFrame in parallel based on a specified function.

    Args:
        func (function): The function to apply to each row of the DataFrame. It should accept arguments that include values from the DataFrame.
        *args: The arguments to pass to the function, which should include the DataFrameto filter and the additional parameters required by `func`.

    Returns:
        pandas.DataFrame: A DataFrame containing only the rows where the function returns True.

    Examples:
    >>> import pandas as pd
    >>> data = {'date': ["2024-06-23", "2024-07-01", "2024-06-01", "2024-06-30", "2023-12-31"],
    ...         'value': [5, 15, 10, 5, 4]}
    >>> df = pd.DataFrame(data)
    >>> date_range = ("2024-06-01", "2024-06-30")
    >>> filter_data_parallely(filter_data_by_date, df, date_range)
             date  value
    0  2024-06-23      5
    1  2024-06-01     10
    2  2024-06-30      5

    >>> value_range = (1, 10)
    >>> filter_data_parallely(filter_data_by_value, df, value_range)
             date  value
    0  2024-06-23      5
    1  2024-06-01     10
    2  2024-06-30      5
    3  2023-12-31      4
    """
    df = args[0]
    chunks = chunk_data(df)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(filter_data_serially, func, chunk, args[1]) for chunk in chunks]
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(e)

    results = pd.concat(results)
    results.reset_index(drop=True, inplace=True)

    return results