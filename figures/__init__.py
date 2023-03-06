import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from data import BreastCancer, SpliceJunction, CensusIncome
from experiments import TEST_RATIO

mpl.use('TkAgg')  # !IMPORTANT

PATH = Path(__file__).parents[0]


def plot_accuracy_distributions(results: list[pd.DataFrame], dataset: BreastCancer or SpliceJunction or CensusIncome,
                                exp_type: str, drop_percentage: int, steps: int, predictor_name: str, metric: str):
    """
    Generate the box plots af the accuracy distributions.
    For each result in results, select the accuracy column and plot it as a box plot.
    The expected number of results is steps.
    :param results: A list of dataframes containing the results of the experiments.
    :param dataset: The dataset object containing all main information.
    :param exp_type: The type of the experiment.
    :param drop_percentage: The percentage of the dataset to drop.
    :param steps: The number of steps.
    :param predictor_name: The name of the predictor.
    :param metric: The metric used to evaluate the predictor.
    """

    data_size = dataset.size * (1 - TEST_RATIO)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.boxplot([result[metric] for result in results], labels=[str(i) for i in range(1, steps + 1)], widths=0.9)
    plt.title(dataset.name.capitalize().replace('-', ' ') + ' ' + predictor_name + ' ' + metric + ' distributions')
    plt.xlabel('Cardinality of the training set')
    plt.ylabel(metric.capitalize() + ' on test set')
    drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
    drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
    labels = [y + "\n" + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
    ax.set_xticks(np.arange(1, steps + 1, 1), labels)
    if not os.path.exists(PATH / exp_type):
        os.makedirs(PATH / exp_type)
    if not os.path.exists(PATH / (exp_type + os.sep + dataset.name)):
        os.makedirs(PATH / (exp_type + os.sep + dataset.name))
    if not os.path.exists(PATH / (exp_type + os.sep + dataset.name + os.sep + predictor_name)):
        os.makedirs(PATH / (exp_type + os.sep + dataset.name + os.sep + predictor_name))
    plt.savefig(PATH / (exp_type + os.sep + dataset.name + os.sep + predictor_name + os.sep + metric + '-distributions.svg'))


def plot_distributions_comparison(data1: list[pd.DataFrame], data2: list[pd.DataFrame],
                                  dataset: BreastCancer or SpliceJunction or CensusIncome,
                                  drop_percentage: int, steps: int, predictor_name1: str, predictor_name2: str,
                                  metric: str):
    """
    Generate the box plots af the accuracy distributions.
    For each result in results, select the accuracy column and plot it as a box plot.
    The expected number of results is steps.
    :param data1: A list of dataframes containing the results of the experiments.
    :param data2: A list of dataframes containing the results of the experiments.
    :param dataset: The dataset object containing all main information.
    :param drop_percentage: The percentage of the dataset to drop.
    :param steps: The number of steps.
    :param predictor_name1: The name of the predictor.
    :param predictor_name2: The name of the predictor.
    :param metric: The metric used to evaluate the predictor.
    """

    main_color1 = 'blue'
    border_color1 = 'royalblue'
    main_color2 = 'red'
    border_color2 = 'pink'
    data_size = dataset.size * (1 - TEST_RATIO)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    distributions1 = [data[metric] for data in data1]
    distributions2 = [data[metric] for data in data2]
    b1 = ax.boxplot(distributions1,
                    positions=np.arange(1, steps + 1, 1),
                    labels=[str(i) for i in range(1, steps + 1)],
                    widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor=border_color1, color=main_color1),
                    capprops=dict(color=main_color1),
                    whiskerprops=dict(color=main_color1),
                    flierprops=dict(color=main_color1, markeredgecolor=main_color1),
                    medianprops=dict(color=main_color1), )
    b2 = ax.boxplot(distributions2,
                    positions=np.arange(1.5, steps + 1.5, 1),
                    labels=[str(i) for i in range(1, steps + 1)],
                    widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor=border_color2, color=main_color2),
                    capprops=dict(color=main_color2),
                    whiskerprops=dict(color=main_color2),
                    flierprops=dict(color=main_color2, markeredgecolor=main_color2),
                    medianprops=dict(color=main_color2), )
    plt.title(dataset.name.capitalize().replace('-', ' ') + ' ' + predictor_name1 + ' vs ' + predictor_name2 + ' ' + metric + ' distributions')
    plt.xlabel('Cardinality of the training set')
    plt.ylabel(metric.capitalize() + ' on test set')
    drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
    drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
    labels = [y + "\n" + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
    ax.set_xticks(np.arange(1.25, steps + 1.25, 1), labels)
    plt.legend([b1["boxes"][0], b2["boxes"][0]], [predictor_name1, predictor_name2], loc='upper right')
    if not os.path.exists(PATH / dataset.name):
        os.makedirs(PATH / dataset.name)
    plt.savefig(PATH / (dataset.name + os.sep + predictor_name1 + '-' + predictor_name2 + '-' + metric + '-distributions.svg'))


def plot_average_accuracy_curves(experiments: list[list[pd.DataFrame]], dataset: BreastCancer or SpliceJunction or CensusIncome,
                                 drop_percentage: int, steps: int, predictor_names: list[str], metric: str):
    """
    Generate the average accuracy curves.
    :param experiments: A list of lists of dataframes containing the results of the experiments.
    :param dataset: The dataset object containing all main information.
    :param drop_percentage: The percentage of the dataset to drop.
    :param steps: The number of steps.
    :param predictor_names: The names of the predictors.
    :param metric: The metric used to evaluate the predictor.
    """

    data_size = dataset.size * (1 - TEST_RATIO)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    predictor_names.insert(0, 'uneducated')
    for i in range(len(experiments)):
        curve = [np.mean(distribution[metric]) for distribution in experiments[i]]  # means of the distributions
        # std_devs = [np.std(distribution[metric]) for distribution in experiments[i]]  # std devs of the distributions
        ax.plot(np.arange(1, steps + 1, 1), curve, label=predictor_names[i], linewidth=2)
        # ax.fill_between(np.arange(1, steps + 1, 1), np.array(curve) - np.array(std_devs),
        #                 np.array(curve) + np.array(std_devs), alpha=0.2)
    plt.title(dataset.name.capitalize().replace('-', ' ') + ' ' + metric + ' average curves')
    plt.xlabel('Cardinality of the training set')
    plt.ylabel(metric.capitalize() + ' on test set')
    drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
    drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
    labels = [y + "\n" + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
    ax.set_xticks(np.arange(1, steps + 1, 1), labels)
    plt.legend(loc='upper right')
    if not os.path.exists(PATH / dataset.name):
        os.makedirs(PATH / dataset.name)
    plt.savefig(PATH / (dataset.name + os.sep + '-'.join(predictor_names) + '-' + metric + '-average-curves.svg'))
