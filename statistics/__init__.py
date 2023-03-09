from pathlib import Path
import numpy as np
import pandas as pd
from data import BreastCancer, SpliceJunction, CensusIncome

PATH = Path(__file__).parents[0]


def apply_noise(data: pd.DataFrame, mu: float, sigma: float, dataset_name: str, seed: int) -> pd.DataFrame:
    """
    Apply noise to the data.
    :param data: the dataset
    :param mu: center of the distribution
    :param sigma: standard deviation of the distribution
    :param dataset_name:
    :param seed: the seed for the random number generator
    :return:
    """
    if dataset_name == BreastCancer.name:
        return _apply_noise_to_breast_cancer(data, mu, sigma, seed)
    elif dataset_name == SpliceJunction.name:
        return _apply_noise_to_splice_junction(data, mu, sigma, seed)
    elif dataset_name == CensusIncome.name:
        return _apply_noise_to_census_income(data, mu, sigma, seed)
    else:
        raise ValueError('The dataset name is not valid.')


def _add_noise_to_ordinal_feature(feature: str, data: pd.DataFrame, mu: float, sigma: float):
    """
    Add noise to an ordinal feature.
    :param feature: the feature to add noise to
    :param data: the dataset
    :param mu: center of the distribution
    :param sigma: standard deviation of the distribution
    :return:
    """
    maximum_value = data[feature].max()
    minimum_value = data[feature].min()
    data[feature] = data[feature].apply(lambda j: round(j + np.random.normal(mu, sigma)))
    data[feature] = data[feature].apply(lambda j: maximum_value if j > maximum_value else j)
    data[feature] = data[feature].apply(lambda j: minimum_value if j < minimum_value else j)


def _apply_noise_to_breast_cancer(data: pd.DataFrame, mu: float, sigma: float, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    data_copy = data.copy()
    for feature in data.columns:
        _add_noise_to_ordinal_feature(feature, data_copy, mu, sigma)
    return data_copy


def _apply_noise_to_splice_junction(data: pd.DataFrame, mu: float, sigma: float, seed: int) -> pd.DataFrame:
    data_copy = data.copy()
    # First, a random order is generated for the 4 basis (a, c, g, t).
    # This order is always the same for a given seed.
    # Then, features are grouped into 60 macro-features, each macro-feature has 4 features.
    # For each feature in a macro-feature, if the feature is 1, then the value of the feature is changed according to
    # the noise and the random order. Otherwise, the value is not changed.
    np.random.seed(0)
    basis_order = np.random.choice(['a', 'c', 'g', 't'], 4, replace=False)
    np.random.seed(seed)
    basis_mapping = {k: v for v, k in enumerate(basis_order)}
    original_mapping = {0: 'a', 1: 'c', 2: 'g', 3: 't'}
    for i in range(0, 60):
        new_values_i = []
        for j in range(0, 4):
            feature = data.columns[i * 4 + j]
            new_values = data_copy[feature].apply(lambda k: round(basis_mapping[original_mapping[j]] + np.random.normal(mu, sigma)) if k == 1 else np.nan)
            new_values = new_values.apply(lambda k: 3 if k > 3 else k)
            new_values = new_values.apply(lambda k: 0 if k < 0 else k)
            new_values_i.append(new_values)
        new_values_i = pd.DataFrame(new_values_i).T
        for j in range(0, 4):
            feature = data.columns[i * 4 + j]
            data_copy[feature] = new_values_i.apply(lambda k: 1 if basis_mapping[original_mapping[j]] in list(k) else 0, axis=1)
    return data_copy


def _apply_noise_to_census_income(data: pd.DataFrame, mu: float, sigma: float, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    data_copy = data.copy()
    for integer_feature in CensusIncome.integer_features:
        data_copy[integer_feature] = data_copy[integer_feature].apply(lambda j: round(j + np.random.normal(mu, sigma)))
        if integer_feature == "Age":
            data_copy[integer_feature] = data_copy[integer_feature].apply(lambda j: 0 if j < 0 else j)
        elif integer_feature == "HoursPerWeek":
            data_copy[integer_feature] = data_copy[integer_feature].apply(lambda j: 0 if j < 0 else j)
            data_copy[integer_feature] = data_copy[integer_feature].apply(lambda j: 99 if j > 99 else j)
    for binary_feature in CensusIncome.binary_features:
        data_copy[binary_feature] = data_copy[binary_feature].apply(lambda j: round(j + np.random.normal(mu, sigma)))
        data_copy[binary_feature] = data_copy[binary_feature].apply(lambda j: 1 if j > 1 else j)
        data_copy[binary_feature] = data_copy[binary_feature].apply(lambda j: 0 if j < 0 else j)
    for ordinal_feature in CensusIncome.ordinal_features:
        _add_noise_to_ordinal_feature(ordinal_feature, data_copy, mu, sigma)
    for nominal_feature in CensusIncome.nominal_features:
        values = [f for f in data_copy.columns if f.startswith(nominal_feature)]
        original_mapping = {k: v for k, v in enumerate(values)}
        np.random.seed(0)
        new_order = np.random.choice(values, len(values), replace=False)
        new_mapping = {k: v for v, k in enumerate(new_order)}
        inverse_new_mapping = {v: k for k, v in new_mapping.items()}
        indices = data_copy[values].apply(lambda j: np.argmax(j), axis=1)
        np.random.seed(seed)
        indices = indices.apply(lambda j: round(new_mapping[original_mapping[j]] + np.random.normal(mu, sigma)))
        indices = indices.apply(lambda j: len(values) - 1 if j > len(values) - 1 else j)
        indices = indices.apply(lambda j: 0 if j < 0 else j)
        for value in values:
            data_copy[value] = indices.apply(lambda j: 1 if inverse_new_mapping[j] == value else 0)
    return data_copy
