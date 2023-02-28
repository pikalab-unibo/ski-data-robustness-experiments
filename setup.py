import distutils.cmd
import os

import pandas as pd
from psyki.logic.prolog import TuProlog
from psyki.ski import Injector
from setuptools import find_packages, setup
from tensorflow.python.keras.metrics import Precision, Recall
from data import load_splice_junction_dataset, SpliceJunction, load_breast_cancer_dataset, BreastCancer, \
    load_census_income_dataset, CensusIncome
from experiments import generate_neural_network_breast_cancer
from knowledge import PATH as KNOWLEDGE_PATH


class LoadDatasets(distutils.cmd.Command):
    description = 'download necessary datasets for the experiments'
    user_options = [('features=', 'f', 'binarize the features of the datasets ([y]/n)'),
                    ('output=', 'o', 'convert class string name into numeric indices ([y]/n)')]
    binary_f = False
    numeric_out = False
    features = 'y'
    output = 'y'

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        self.binary_f = self.features.lower() == 'y'
        self.numeric_out = self.output.lower() == 'y'

    def run(self) -> None:
        splice_dataset = load_splice_junction_dataset(self.binary_f, self.numeric_out)
        splice_dataset.to_csv(SpliceJunction.file_name, index=False)
        breast_dataset = load_breast_cancer_dataset(self.numeric_out)
        breast_dataset.to_csv(BreastCancer.file_name, index=False)
        census_dataset = load_census_income_dataset(self.binary_f, self.numeric_out)
        census_dataset.to_csv(CensusIncome.file_name, index=False)


class RunExperimentsWithDataDrop(distutils.cmd.Command):
    description = 'run experiments'
    user_options = []
    metrics = ['accuracy', Precision(), Recall()]
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    population_size = 30

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from experiments import experiment_with_data_drop
        breast_cancer_dataset = pd.read_csv(BreastCancer.file_name, header=0, sep=",", encoding='utf8')
        uneducated = generate_neural_network_breast_cancer(self.metrics)
        # knowledge = TuProlog.from_file(KNOWLEDGE_PATH / BreastCancer.knowledge_file_name).formulae
        # feature_mapping = {k: v for v, k in enumerate(breast_cancer_dataset.columns[:-1])}
        # kins = Injector.kins(uneducated, feature_mapping)
        # educated = kins.inject(knowledge)
        # educated.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        experiment_with_data_drop(breast_cancer_dataset, uneducated, 'breast_cancer', 'uneducated', self.population_size, self.metrics)


class GeneratePlots(distutils.cmd.Command):
    description = 'generate plots'
    user_options = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        from statistics import plot_accuracy_distributions
        from results.drop import PATH as DROP_RESULT_PATH
        results = []
        for file in os.listdir(DROP_RESULT_PATH):
            if file.endswith('.csv') and file.startswith('breast_cancer_uneducated'):
                results.append(pd.read_csv(DROP_RESULT_PATH / file, header=0, sep=",", encoding='utf8'))
        plot_accuracy_distributions(results)


setup(
    name='Experiments on the robustness of symbolic knowledge injection techniques w.r.t. data quality degradation',
    description='SKI QoS experiments',
    license='Apache 2.0 License',
    url='https://github.com/pikalab-unibo/ski-qos-jaamas-experiments-2022',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='symbolic knowledge injection, ski, symbolic ai',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'psyki>=0.2.19',
        'psyke>=0.3.3.dev13',
        'tensorflow>=2.7.0',
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
    ],  # Optional
    zip_safe=False,
    cmdclass={
        'load_datasets': LoadDatasets,
        'run_experiments_with_data_drop': RunExperimentsWithDataDrop,
        'generate_plots': GeneratePlots,
    },
)
