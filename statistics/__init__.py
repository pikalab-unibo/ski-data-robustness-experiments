from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path(__file__).parents[0]


def apply_noise(data: pd.DataFrame, mu: float, sigma: float, column_data_types: dict[str, str], seed: int) -> pd.DataFrame:
    """
    Apply noise to the data.
    :param data: the dataset
    :param mu: center of the distribution
    :param sigma: standard deviation of the distribution
    :param column_data_types: the data types of the columns
    :param ratio: the ratio of the data to be affected by the noise
    :param seed: the seed for the random number generator
    :return:
    """
    np.random.seed(seed)
    data_copy = data.copy()
    for k, v in column_data_types.items():
        if v == 'float':
            data_copy[k] = data_copy[k] + np.random.normal(mu, sigma, len(data_copy[k]))
        elif v == 'int':  # also for ordinal categorical data
            maximum_value = data_copy[k].max()
            minimum_value = data_copy[k].min()
            data_copy[k] = data_copy[k].apply(lambda j: round(j + np.random.normal(mu, sigma)))
            data_copy[k] = data_copy[k].apply(lambda j: maximum_value if j > maximum_value else j)
            data_copy[k] = data_copy[k].apply(lambda j: minimum_value if j < minimum_value else j)
        elif v == 'categorical':
            # TODO
            pass
    return data_copy
