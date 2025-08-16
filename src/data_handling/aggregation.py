from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def serial_sum(data):
    """
    Sum all elements of a list serially. 
    Args:
        data (list of int): The list of integers to sum.

    Returns:
        int: The sum of the list elements.

    Example:
    >>> serial_sum([1, 2, 3, 4, 5])
    15
    """
    total_sum = 0

    for num in data:
        total_sum += num

    return total_sum


def parallel_sum(data):
    """
    Sum all elements of a list in parallel. The list is divided into chunks, and each chunk is summed in parallel.

    Args:
        data (list of int): The list of integers to sum.

    Returns:
        int: The sum of the list elements.

    Example:
    >>> parallel_sum([1, 2, 3, 4, 5])
    15
    """
    num_processes = os.cpu_count()
    chunk_size = len(data) // num_processes
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes-1)]
    chunks.append(data[(num_processes-1)*chunk_size:])

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(serial_sum, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]

    total_sum = serial_sum(results)

    return total_sum