from typing import Type, Union
import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from data import BreastCancer, SpliceJunction, CensusIncome
from experiments import TEST_RATIO
from mlxtend.plotting import plot_confusion_matrix

mpl.use('TkAgg')  # !IMPORTANT

PATH = Path(__file__).parents[0]


def _create_missing_directories(path: Path, exp_type: str, dataset: BreastCancer or SpliceJunction or CensusIncome or str):
    if not os.path.exists(path / exp_type):
        os.makedirs(path / exp_type)
    if isinstance(dataset, str):
        if not os.path.exists(path / (exp_type + os.sep + dataset)):
            os.makedirs(path / (exp_type + os.sep + dataset))
    else:
        if not os.path.exists(path / (exp_type + os.sep + dataset.name)):
            os.makedirs(path / (exp_type + os.sep + dataset.name))


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
    plt.ylabel(metric.capitalize() + ' on test set')
    if exp_type == 'drop':
        plt.xlabel('Cardinality of the training set')
        drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
        drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
        labels = [y + "\n" + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels)
    elif exp_type == 'noise':
        plt.xlabel('Noise level (sigma)')
        if dataset.name == SpliceJunction.name:
            ax.set_xticks(np.arange(1, steps + 1, 1), [f'{i / 10}' for i in range(0, steps)])
        else:
            ax.set_xticks(np.arange(1, steps + 1, 1), [f'{i}' for i in range(0, steps)])
    elif exp_type == 'mix':
        plt.xlabel(r'Cardinality of the training set ($\left\|\cdot\right\|$) and noise level ($\sigma$)')
        drop_percentage_labels = [r'$\left\|\cdot\right\|$ = ' \
                                  r'{}%'.format(100 - i) for i in range(0,
                                                                        drop_percentage * steps,
                                                                        drop_percentage)]
        if dataset.name == SpliceJunction.name:
            noise_value_labels = [r'$\sigma$={}'.format(i/10) for i in range(steps)]
        else:
            noise_value_labels = [r'$\sigma$={}'.format(i) for i in range(steps)]
        labels = [y + " & " + x for x, y in zip(drop_percentage_labels, noise_value_labels)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels, rotation=90)
    elif exp_type == 'label_flip':
        plt.xlabel(r'Flipping probability $P_f$')
        labels = [r'$P_f$ = {}%'.format(100*(0.9 / steps) * i) for i in range(0, steps)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels, rotation=80)
    _create_missing_directories(PATH, exp_type, dataset)
    if not os.path.exists(PATH / (exp_type + os.sep + dataset.name + os.sep + predictor_name)):
        os.makedirs(PATH / (exp_type + os.sep + dataset.name + os.sep + predictor_name))
    plt.savefig(
        PATH / (exp_type + os.sep + dataset.name + os.sep + predictor_name + os.sep + metric + '-distributions.svg'))


def plot_distributions_comparison(data1: list[pd.DataFrame], data2: list[pd.DataFrame],
                                  dataset: BreastCancer or SpliceJunction or CensusIncome, exp_type: str,
                                  drop_percentage: int, steps: int, predictor_name1: str, predictor_name2: str,
                                  metric: str):
    """
    Generate the box plots af the accuracy distributions.
    For each result in results, select the accuracy column and plot it as a box plot.
    The expected number of results is steps.
    :param data1: A list of dataframes containing the results of the experiments.
    :param data2: A list of dataframes containing the results of the experiments.
    :param dataset: The dataset object containing all main information.
    :param exp_type: The type of the experiment.
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
    plt.title(dataset.name.capitalize().replace('-',
                                                ' ') + ' ' + predictor_name1 + ' vs ' + predictor_name2 + ' ' + metric + ' distributions')
    plt.ylabel(metric.capitalize() + ' on test set')
    if exp_type == 'drop':
        plt.xlabel('Cardinality of the training set')
        drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
        drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
        labels = [y + "\n" + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
        ax.set_xticks(np.arange(1.25, steps + 1.25, 1), labels)
    elif exp_type == 'noise':
        plt.xlabel(r'Noise level ($\sigma$)')
        if dataset.name == SpliceJunction.name:
            ax.set_xticks(np.arange(1.25, steps + 1.25, 1), [f'{i / 10}' for i in range(0, steps)])
        else:
            ax.set_xticks(np.arange(1.25, steps + 1.25, 1), [f'{i}' for i in range(0, steps)])
    elif exp_type == 'mix':
        plt.xlabel(r'Cardinality of the training set ($\left\|\cdot\right\|$) and noise level ($\sigma$)')
        drop_percentage_labels = [r'$\left\|\cdot\right\|$ = ' \
                                  r'{}%'.format(100 - i) for i in range(0,
                                                                        drop_percentage * steps,
                                                                        drop_percentage)]
        if dataset.name == SpliceJunction.name:
            noise_value_labels = [r'$\sigma$={}'.format(i/10) for i in range(steps)]
        else:
            noise_value_labels = [r'$\sigma$={}'.format(i) for i in range(steps)]
        labels = [y + " & " + x for x, y in zip(drop_percentage_labels, noise_value_labels)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels, rotation=90)
    elif exp_type == 'label_flip':
        plt.xlabel(r'Flipping probability $P_f$')
        labels = [r'$P_f$ = {}%'.format(100*(0.9 / steps) * i) for i in range(0, steps)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels, rotation=80)
    plt.legend([b1["boxes"][0], b2["boxes"][0]], [predictor_name1, predictor_name2], loc='upper right')
    _create_missing_directories(PATH, exp_type, dataset)
    plt.savefig(PATH / (
            exp_type + os.sep + dataset.name + os.sep + predictor_name1 + '-' + predictor_name2 + '-' + metric + '-distributions.svg'))


def plot_average_accuracy_curves(experiments: list[list[pd.DataFrame]],
                                 dataset: BreastCancer or SpliceJunction or CensusIncome,
                                 exp_type: str, drop_percentage: int, steps: int, predictor_names: list[str],
                                 metric: str):
    """
    Generate the average accuracy curves.
    :param experiments: A list of lists of dataframes containing the results of the experiments.
    :param dataset: The dataset object containing all main information.
    :param exp_type: The type of the experiment.
    :param drop_percentage: The percentage of the dataset to drop.
    :param steps: The number of steps.
    :param predictor_names: The names of the predictors.
    :param metric: The metric used to evaluate the predictor.
    """

    lines = {'uneducated': 'solid',
             'kbann': (0, (3, 5, 1, 5, 1, 5)),
             'kill': (0, (3, 5, 1, 5)),
             'kins': (0, (5, 10))}
    markers = {'uneducated': 'o',
               'kbann': 'v',
               'kill': '^',
               'kins': 's'}
    colors = {'uneducated': 'red',
              'kbann': 'blue',
              'kill': 'green',
              'kins': 'black'}
    fontsizes = {'title': 19,
                 'legend': 22,
                 'axis': 25,
                 'ticks': 20, }
    legend_font = font_manager.FontProperties(style='normal', size=fontsizes['legend'])

    data_size = dataset.size * (1 - TEST_RATIO)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    predictor_names.insert(0, 'uneducated')
    for i in range(len(experiments)):
        if metric == 'f1':
            precisions = [distribution['precision'] for distribution in experiments[i]]
            recalls = [distribution['recall'] for distribution in experiments[i]]
            curve = [np.mean((2 * p * r) / (p + r)) for p, r in zip(precisions, recalls)]
        else:
            curve = [np.mean(distribution[metric]) for distribution in experiments[i]]  # means of the distributions
        # std_devs = [np.std(distribution[metric]) for distribution in experiments[i]]  # std devs of the distributions
        ax.plot(np.arange(1, steps + 1, 1), curve,
                # linestyle=lines[predictor_names[i]],
                marker=markers[predictor_names[i]],
                markersize=10,
                color=colors[predictor_names[i]],
                label=predictor_names[i].upper(),
                linewidth=3)
        # ax.fill_between(np.arange(1, steps + 1, 1), np.array(curve) - np.array(std_devs),
        #                 np.array(curve) + np.array(std_devs), alpha=0.2)
    # plt.title(dataset.name.capitalize().replace('-', ' ') + ' ' + metric + ' average curves',
    #           fontsize=fontsizes['title'])
    plt.ylabel(metric.capitalize() + ' on test set',
               fontsize=fontsizes['axis'])
    if exp_type == 'drop':
        plt.xlabel('Cardinality of the training set',
                   fontsize=fontsizes['axis'])
        drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
        drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
        # labels = [y + "\n" + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
        labels = [y + " " + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels,
                      fontsize=fontsizes['ticks'], rotation=45)
        plt.legend(loc='lower left', prop=legend_font)
    elif exp_type == 'noise':
        plt.xlabel(r'Noise level ($\sigma$)',
                   fontsize=fontsizes['axis'])
        if dataset.name == SpliceJunction.name:
            ax.set_xticks(np.arange(1, steps + 1, 1), [f'{i / 10}' for i in range(0, steps)],
                          fontsize=fontsizes['ticks'])
        else:
            ax.set_xticks(np.arange(1, steps + 1, 1), [f'{i}' for i in range(0, steps)],
                          fontsize=fontsizes['ticks'])
        plt.legend(loc='upper right', prop=legend_font)
    elif exp_type == 'mix':
        plt.xlabel(r'Cardinality of the training set ($\left\|\cdot\right\|$) and noise level ($\sigma$)',
                   fontsize=fontsizes['axis'])
        drop_percentage_labels = [r'$\left\|\cdot\right\|$ = ' \
                                  r'{}%'.format(100 - i) for i in range(0,
                                                                        drop_percentage * steps,
                                                                        drop_percentage)]
        if dataset.name == SpliceJunction.name:
            noise_value_labels = [r'$\sigma$={}'.format(i/10) for i in range(steps)]
        else:
            noise_value_labels = [r'$\sigma$={}'.format(i) for i in range(steps)]
        labels = [y + " & " + x for x, y in zip(drop_percentage_labels, noise_value_labels)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels,
                      fontsize=fontsizes['ticks'], rotation=90)
        ax.set_yscale('log')
        plt.legend(loc='lower right', prop=legend_font)
    elif exp_type == 'label_flip':
        plt.xlabel(r'Flipping probability $P_f$', fontsize=fontsizes['axis'])
        labels = [r'$P_f$ = {}%'.format(100*(0.9 / steps) * i) for i in range(0, steps)]
        ax.set_xticks(np.arange(1, steps + 1, 1), labels,
                      fontsize=fontsizes['ticks'], rotation=80)
        plt.legend(loc='upper left', prop=legend_font)
    plt.yticks(fontsize=fontsizes['ticks'])
    plt.tight_layout()
    _create_missing_directories(PATH, exp_type, dataset)
    plt.savefig(PATH / (exp_type + os.sep + dataset.name + os.sep + '-'.join(
        predictor_names) + '-' + metric + '-average-curves.svg'))
    plt.savefig(PATH / (exp_type + os.sep + dataset.name + os.sep + '-'.join(
        predictor_names) + '-' + metric + '-average-curves.pdf'))


def plot_divergences_distributions(experiments: dict[Type[Union[BreastCancer, SpliceJunction, CensusIncome]], list],
                                   exp_type: str, drop_percentage: int, steps: int, ):
    """
    Generate the average accuracy curves.
    :param experiments: A dictionary of lists of dataframes containing the results of the experiments.
    :param exp_type: The type of the experiment.
    :param drop_percentage: The percentage of the dataset to drop.
    :param steps: The number of steps.
    """

    lines = {'breast-cancer': (0, (3, 5, 1, 5, 1, 5)),
             'splice-junction': (0, (3, 5, 1, 5)),
             'census-income': (0, (5, 10))}
    markers = {'breast-cancer': 'v',
               'splice-junction': '^',
               'census-income': 's'}
    colors = {'breast-cancer': 'blue',
              'splice-junction': 'green',
              'census-income': 'black'}
    fontsizes = {'title': 19,
                 'legend': 22,
                 'axis': 25,
                 'ticks': 20, }
    legend_font = font_manager.FontProperties(style='normal', size=fontsizes['legend'])

    datasets = list(experiments.keys())
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    def reject_outliers(data, m=2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else np.zeros(len(d))
        return data[s < m]

    for dataset in datasets:
        dataset_divergences = experiments[dataset]
        dataset_name = dataset.name
        data_size = dataset.size * (1 - TEST_RATIO)
        curve = []
        stds = []
        for i in range(len(dataset_divergences)):
            curve.append(np.mean(reject_outliers(dataset_divergences[i]['divergence'])))  # means of the distributions
        curve[0] += 1e-15
        ax.plot(np.arange(1, steps + 1, 1),
                curve,
                # linestyle=lines[dataset.name],
                marker=markers[dataset.name],
                markersize=10,
                color=colors[dataset.name],
                label=dataset.name.upper(),
                linewidth=3)
        plt.ylabel('KL divergence', fontsize=fontsizes['axis'])
        if exp_type == 'drop':
            plt.xlabel('Cardinality of the training set',
                       fontsize=fontsizes['axis'])
            drop_percentage_labels = [f'({100 - i}%)' for i in range(0, drop_percentage * steps, drop_percentage)]
            drop_value_labels = [f'{round(data_size * (1 - (i * drop_percentage / 100)))}' for i in range(steps)]
            labels = [y + " " + x for x, y in zip(drop_percentage_labels, drop_value_labels)]
            ax.set_xticks(np.arange(1, steps + 1, 1), labels,
                          fontsize=fontsizes['ticks'], rotation=45)
            ax.set_yscale('log')
            plt.legend(loc='lower right', prop=legend_font)
        elif exp_type == 'noise':
            plt.xlabel(r'Noise level ($\sigma$)',
                       fontsize=fontsizes['axis'])
            if dataset.name == SpliceJunction.name:
                ax.set_xticks(np.arange(1, steps + 1, 1), [f'{i / 10}' for i in range(0, steps)],
                              fontsize=fontsizes['ticks'])
            else:
                ax.set_xticks(np.arange(1, steps + 1, 1), [f'{i}' for i in range(0, steps)],
                              fontsize=fontsizes['ticks'])
            plt.legend(loc='lower right', prop=legend_font)
        elif exp_type == 'mix':
            plt.xlabel(r'Cardinality of the training set ($\left\|\cdot\right\|$) and noise level ($\sigma$)',
                       fontsize=fontsizes['axis'])
            drop_percentage_labels = [r'$\left\|\cdot\right\|$ = ' \
                                      r'{}%'.format(100 - i) for i in range(0,
                                                                            drop_percentage * steps,
                                                                            drop_percentage)]
            if dataset.name == SpliceJunction.name:
                noise_value_labels = [r'$\sigma$={}'.format(i/10) for i in range(steps)]
            else:
                noise_value_labels = [r'$\sigma$={}'.format(i) for i in range(steps)]
            labels = [y + " & " + x for x, y in zip(drop_percentage_labels, noise_value_labels)]
            ax.set_xticks(np.arange(1, steps + 1, 1), labels,
                          fontsize=fontsizes['ticks'], rotation=90)
            ax.set_yscale('log')
            plt.legend(loc='lower right', prop=legend_font)
        elif exp_type == 'label_flip':
            plt.xlabel(r'Flipping probability $P_f$', fontsize=fontsizes['axis'])
            labels = [r'$P_f$ = {}%'.format(100*(0.9 / steps) * i) for i in range(0, steps)]
            ax.set_xticks(np.arange(1, steps + 1, 1), labels,
                          fontsize=fontsizes['ticks'], rotation=80)
            plt.legend(loc='upper left', prop=legend_font)
    plt.yticks(fontsize=fontsizes['ticks'])
    plt.tight_layout()
    _create_missing_directories(PATH, exp_type, 'divergences')
    plt.savefig(PATH / (exp_type + os.sep + 'divergences' + os.sep + 'KL-' + exp_type + '-average-curves.svg'))
    plt.savefig(PATH / (exp_type + os.sep + 'divergences' + os.sep + 'KL-' + exp_type + '-average-curves.pdf'))


def plot_cm(data: np.ndarray, class_names: list[str], dataset_name: str):
    fig, ax = plot_confusion_matrix(data, show_absolute=True, show_normed=True, colorbar=True, class_names=class_names)
    fig.savefig(PATH / (dataset_name + '.svg'))
