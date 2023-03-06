from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path(__file__).parents[0]


def apply_noise(data: pd.DataFrame, mu: float, sigma: float, column_data_types: dict[str, str], ratio: float,
                seed: int) -> pd.DataFrame:
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
    noise_indices = np.random.choice(data_copy.index, int(data_copy.shape[0] * ratio), replace=False)
    for k, v in column_data_types.items():
        if v == 'float':
            for noise_index in noise_indices:
                data_copy[k][noise_index] = data_copy[k][noise_index] + np.random.normal(mu, sigma, 1)[0]
        elif v == 'int':  # also for ordinal categorical data
            maximum_value = data_copy[k].max()
            minimum_value = data_copy[k].min()
            for index in noise_indices:
                data_copy[k][index] = round((data_copy[k][index] + np.random.normal(mu, sigma, 1))[0])
                if data_copy[k][index] > maximum_value:
                    data_copy[k][index] = maximum_value
                elif data_copy[k][index] < minimum_value:
                    data_copy[k][index] = minimum_value
        elif v == 'categorical':
            # TODO
            pass
    return data_copy
