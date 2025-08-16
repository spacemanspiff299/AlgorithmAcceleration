from data_handling import serial_sum, parallel_sum
import time
import unittest
import random


def generate_random_list(size):
    """
    Generate a list of random integers in the range [1, 100] with a specified size.

    Args:
        size (int): Length of the list.

    Returns:
        list: A list of random integers of the specified size.

    Example:
    >>> len(generate_random_list(5)) == 5
    True
    """
    random.seed(0)
    return [random.randint(1, 100) for _ in range(size)]


class TestRandomListGenerator(unittest.TestCase):
    def test_generate_random_list_size(self):
        size = 10
        result = generate_random_list(size)
        self.assertEqual(len(result), size)


class TestSerialSum(unittest.TestCase):
    def test_serial_sum_empty(self):
        self.assertEqual(serial_sum([]), 0)

    def test_serial_sum_single_element(self):
        self.assertEqual(serial_sum([42]), 42)

    def test_serial_sum_multiple_elements(self):
        data = [1, 2, 3, 4, 5]
        self.assertEqual(serial_sum(data), 15)


class TestParallelSum(unittest.TestCase):
    def test_parallel_sum_empty(self):
        self.assertEqual(parallel_sum([]), 0)

    def test_parallel_sum_single_element(self):
        self.assertEqual(parallel_sum([42]), 42)

    def test_parallel_sum_multiple_elements(self):
        data = [1, 2, 3, 4, 5]
        self.assertEqual(parallel_sum(data), 15)

    def test_parallel_sum_large_list(self):
        data = generate_random_list(1000)
        self.assertEqual(parallel_sum(data), sum(data))

    def test_parallel_sum_performance(self):
        data = generate_random_list(5000)
        start_serial = time.perf_counter()
        serial_result = serial_sum(data)
        end_serial = time.perf_counter()
        serial_time = end_serial - start_serial

        start_parallel = time.perf_counter()
        parallel_result = parallel_sum(data)
        end_parallel = time.perf_counter()
        parallel_time = end_parallel - start_parallel

        self.assertEqual(serial_result, parallel_result)