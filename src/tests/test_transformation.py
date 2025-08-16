import unittest
import random
import numpy as np
from data_handling import MinMaxScaler, Normalize, transform_data_serially, transform_data_parallely, transform_data


def generate_random_list(size):
    '''
    Generates a list of random integers between 1 and 100.

    Args:
        size (int): The number of random integers to generate.

    Returns:
        list: A list containing `size` random integers between 1 and 100.

    Example:
    >>> random.seed(0)
    >>> generate_random_list(5)
    [50, 98, 54, 6, 34]
    '''
    return [random.randint(1, 100) for _ in range(size)]


class TestTransformFunctions(unittest.TestCase):

    def setUp(self):
        self.data = generate_random_list(10000)
        self.min = min(self.data)
        self.max = max(self.data)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)

    def test_transform_data(self):
        scaledData = transform_data(MinMaxScaler, self.data, (self.min, self.max))
        normalizedData = transform_data(Normalize, self.data, self.mean, self.std)
        self.assertTrue(len(scaledData), len(self.data))
        self.assertTrue(len(normalizedData), len(self.data))
        self.assertTrue(all(0 <= x <= 1 for x in scaledData))
        self.assertAlmostEqual(0.0, np.mean(normalizedData), delta=1e-6)
        self.assertAlmostEqual(1.0, np.std(normalizedData), delta=1e-6)

    def test_transform_data_serially_vs_parallely(self):
        result_serial = transform_data_serially(MinMaxScaler, self.data)
        result_parallel = transform_data_parallely(MinMaxScaler, self.data)
        self.assertEqual(len(result_serial), len(result_parallel))
        self.assertEqual(result_serial, result_parallel)