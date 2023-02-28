from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from figures import PATH as FIGURE_PATH

mpl.use('TkAgg')  # !IMPORTANT

PATH = Path(__file__).parents[0]


def plot_accuracy_distributions(results: list[pd.DataFrame]):
    """
    Generate the box plots af the accuracy distributions.
    For each result in results, select the accuracy column and plot it as a box plot.
    The expected number of results is 10.
    :param results: A list of dataframes containing the results of the experiments.
    """

    plt.figure(figsize=(12, 8))
    plt.title('Accuracy distributions')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 15, 1))
    plt.boxplot([result['accuracy'] for result in results], labels=[str(i) for i in range(1, 16)])
    plt.savefig(FIGURE_PATH / 'accuracy-distributions.pdf', transparent=True)
