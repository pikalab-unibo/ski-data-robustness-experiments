from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    if ratio == 0:
        x, y = data_copy, None
    else:
        x, y = train_test_split(data_copy, train_size=ratio, random_state=seed, stratify=data_copy.iloc[:, -1])
    for k, v in column_data_types.items():
        if v == 'float':
            x[k] = x[k] + np.random.normal(mu, sigma, len(x[k]))
        elif v == 'int':  # also for ordinal categorical data
            maximum_value = x[k].max()
            minimum_value = x[k].min()
            x[k] = x[k].apply(lambda j: round(j + np.random.normal(mu, sigma)))
            x[k] = x[k].apply(lambda j: maximum_value if j > maximum_value else j)
            x[k] = x[k].apply(lambda j: minimum_value if j < minimum_value else j)
        elif v == 'categorical':
            # TODO
            pass
    if y is not None:
        data_copy = pd.concat((x, y))
    else:
        data_copy = x
    return data_copy
