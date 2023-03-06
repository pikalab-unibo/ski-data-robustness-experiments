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
    if ratio == 0:
        x, y = data_copy, None
    else:
        x, y = data_copy.train_test_split(train_size=ratio, random_state=seed, stratify=data_copy.iloc[:, -1])
    for k, v in column_data_types.items():
        if v == 'float':
            for noise_index in noise_indices:
                x[k][noise_index] = x[k][noise_index] + np.random.normal(mu, sigma, 1)[0]
        elif v == 'int':  # also for ordinal categorical data
            maximum_value = x[k].max()
            minimum_value = x[k].min()
            for index in noise_indices:
                x[k][index] = round((x[k][index] + np.random.normal(mu, sigma, 1))[0])
                if x[k][index] > maximum_value:
                    x[k][index] = maximum_value
                elif x[k][index] < minimum_value:
                    x[k][index] = minimum_value
        elif v == 'categorical':
            # TODO
            pass
    if y:
        data_copy = x.join(y)
    else:
        data_copy = x
    return data_copy
