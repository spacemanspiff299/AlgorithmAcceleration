from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os


def MinMaxScaler(x, data_range, feature_range=(0, 1)):
    """
    Scales a value `x` using Min-Max scaling.

    Args:
        x (float): The value to be scaled.
        data_range (tuple): A tuple (data_min, data_max) representing the range of the data.
        feature_range (tuple, optional): A tuple (feature_min, feature_max) representing the desired range of the feature after scaling. Default is (0, 1).

    Returns:
        float: The scaled value of `x`.

    Example:
    >>> MinMaxScaler(50, (0, 100), (0, 1))
    0.5
    """
    data_min, data_max = data_range[0], data_range[1]
    feature_min, feature_max = feature_range[0], feature_range[1]
    x_std = (x - data_min) / (data_max - data_min)
    x_scaled = x_std * (feature_max - feature_min) + feature_min

    return x_scaled


def Normalize(x, mean, std):
    """
    Normalizes a value `x` using z-score normalization.

    Args:
        x (float): The value to be normalized.
        mean (float): The mean of the data.
        std (float): The standard deviation of the data.

    Returns:
        float: The normalized value of `x`.

    Example:
    >>> Normalize(50, 40, 10)
    1.0
    """
    return (x - mean) / std


def transform_data(func, data, *args):
    """
    Applies a transformation function `func` to each element in `data`.

    Args:
        func (function): The transformation function to apply.
        data (list): The list of data elements to transform.
        *args: Additional arguments required by `func`.

    Returns:
        list: A list containing the transformed elements.

    Example:
    >>> transform_data(MinMaxScaler, [50, 98, 54, 6, 34], (0, 100), (0, 1))
    [0.5, 0.98, 0.54, 0.06, 0.34]
    >>> transform_data(Normalize, [50, 98, 54, 6, 34], 40, 10)
    [1.0, 5.8, 1.4, -3.4, -0.6]
    """
    return [func(x, *args) for x in data]


def flatten(matrix):
    """
    Flattens a 2D list into a 1D list.

    Args:
        matrix (list): A 2D list to flatten.

    Returns:
        list: A flattened 1D list.

    Example:
    >>> flatten([[1, 2, 3], [4, 5], [6]])
    [1, 2, 3, 4, 5, 6]
    """
    flat_list = []
    for row in matrix:
        flat_list.extend(row)

    return flat_list


def transform_data_serially(func, data):
    """
    Applies a transformation function `func` to each element in `data` sequentially.

    Args:
        func (function): The transformation function to apply.
        data (list): The list of data elements to transform.

    Returns:
        list: A list containing the transformed elements.

    Example:
    >>> data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> transform_data_serially(MinMaxScaler, data)
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    """
    if func == MinMaxScaler:
        data_range = (min(data), max(data))
        args = (data_range,)
    elif func == Normalize:
        mean, std = np.mean(data), np.std(data)
        args = (mean, std)

    results = transform_data(func, data, *args)

    return results


def transform_data_parallely(func, data):
    """
    Applies a transformation function `func` to each element in `data` using parallel processing.

    Args:
        func (function): The transformation function to apply.
        data (list): The list of data elements to transform.

    Returns:
        list: A list containing the transformed elements.

    Example:
    >>> data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> transform_data_serially(MinMaxScaler, data)
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    """
    num_processes = os.cpu_count()
    chunk_size = len(data) // num_processes
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes-1)]
    chunks.append(data[(num_processes-1)*chunk_size:])

    if func == MinMaxScaler:
        data_range = (min(data), max(data))
        args = (data_range,)
    elif func == Normalize:
        mean, std = np.mean(data), np.std(data)
        args = (mean, std)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(transform_data, func, chunk, *args) for chunk in chunks]
        results = [future.result() for future in futures]

    results = flatten(results)

    return results