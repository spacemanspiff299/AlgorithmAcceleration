import unittest
import random
import pandas as pd
from datetime import datetime, timedelta
from data_handling import filter_data_by_date, filter_data_by_value, filter_data_serially, filter_data_parallely


def create_simulated_data(num_rows):
    """
    Create a DataFrame with simulated data including random integer values and dates.

    Parameters:
    num_rows (int): The number of rows of data to generate.

    Returns:
    pandas.DataFrame: A DataFrame with columns 'value' and 'date'. 'value' contains random integers between 1 and 100, and 'date' contains dates in the format "YYYY-MM-DD".

    Examples:
    >>> df = create_simulated_data(5)
    >>> len(df) == 5
    True
    >>> df['value'].between(1, 100).all()
    True
    >>> all(isinstance(date, str) and len(date) == 10 for date in df['date'])
    True
    """
    random.seed(0)
    values = [random.randint(1, 100) for _ in range(num_rows)]
    start_date = datetime(2020, 1, 1)
    dates = [(start_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(num_rows)]
    data = {
        'value': values,
        'date': dates
    }
    df = pd.DataFrame(data)

    return df


class TestFunctions(unittest.TestCase):
    def test_create_simulated_data(self):
        df = create_simulated_data(100)
        self.assertEqual(len(df), 100)
        self.assertTrue(df['value'].between(1, 100).all())

    def test_filter_data_serially(self):
        # Create a sample DataFrame
        data = {'date': ["2024-06-23", "2024-07-01", "2024-06-01", "2024-06-30", "2023-12-31"],
                'value': [5, 15, 10, 5, 4]}
        df = pd.DataFrame(data)

        # Test filtering by date
        date_range = ("2024-06-01", "2024-06-30")
        filtered_df_serial = filter_data_serially(filter_data_by_date, df, date_range)
        self.assertEqual(len(filtered_df_serial), 3)  # Expecting 3 rows in the filtered DataFrame

        # Test filtering by value
        value_range = (1, 10)
        filtered_df_serial = filter_data_serially(filter_data_by_value, df, value_range)
        self.assertEqual(len(filtered_df_serial), 4)  # Expecting 4 rows in the filtered DataFrame

    def test_filter_data_parallely(self):
        # Create a sample DataFrame
        data = {'date': ["2024-06-23", "2024-07-01", "2024-06-01", "2024-06-30", "2023-12-31"],
                'value': [5, 15, 10, 5, 4]}
        df = pd.DataFrame(data)

        # Test filtering by date
        date_range = ("2024-06-01", "2024-06-30")
        filtered_df_parallel = filter_data_parallely(filter_data_by_date, df, date_range)
        self.assertEqual(len(filtered_df_parallel), 3)  # Expecting 3 rows in the filtered DataFrame

        # Test filtering by value
        value_range = (1, 10)
        filtered_df_parallel = filter_data_parallely(filter_data_by_value, df, value_range)
        self.assertEqual(len(filtered_df_parallel), 4)  # Expecting 4 rows in the filtered DataFrame

    def test_filter_data(self):
        # Test with a very large DataFrame
        large_df = create_simulated_data(200_000)

        # Serially
        date_range = ("2024-06-01", "2024-06-30")
        filtered_df_serial_large = filter_data_serially(filter_data_by_date, large_df, date_range)

        value_range = (1, 10)
        filtered_df_serial_large = filter_data_serially(filter_data_by_value, large_df, value_range)

        # Parallely
        date_range = ("2024-06-01", "2024-06-30")
        filtered_df_parallel_large = filter_data_parallely(filter_data_by_date, large_df, date_range)

        value_range = (1, 10)
        filtered_df_parallel_large = filter_data_parallely(filter_data_by_value, large_df, value_range)

        # Verify both serial and parallel implementations yield the same results
        pd.testing.assert_frame_equal(filtered_df_serial_large, filtered_df_parallel_large)