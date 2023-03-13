import distutils.cmd
import os
from pathlib import Path
import pandas as pd
from keras.utils.generic_utils import get_custom_objects
from psyki.fuzzifiers.netbuilder import NetBuilder
from psyki.logic.prolog import TuProlog
from psyki.ski import Injector
from setuptools import find_packages, setup
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.python.keras.metrics import Precision, Recall
from data import load_splice_junction_dataset, SpliceJunction, load_breast_cancer_dataset, BreastCancer, \
    load_census_income_dataset, CensusIncome
from experiments import generate_neural_network_breast_cancer, generate_neural_network_census_income, \
    generate_neural_network_splice_junction, SEED
from figures import plot_cm
from knowledge import PATH as KNOWLEDGE_PATH, compute_confusion_matrix
from experiments import experiment_with_data_drop, experiment_with_data_noise, \
    compute_divergence_over_experiments_with_data_noise, compute_divergence_over_experiments_experiment_with_data_drop
from statistics import compute_robustness


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


class RunExperiments(distutils.cmd.Command):
    description = 'run experiments'
    user_options = [('type=', 't', 'type of experiment (d[rop], n[oise])'),
                    ('dataset=', 'd',
                     'dataset to run the experiments on (b[reast cancer], s[plice junction], c[ensus income])'),
                    ('predictor=', 'p', 'predictors to use (u[neducated], kins, kill, kbann)')]
    metrics = ['accuracy', Precision(), Recall()]
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    population_size = 30
    datasets = [BreastCancer, SpliceJunction, CensusIncome]
    predictor_names = ['uneducated', 'kins', 'kill', 'kbann']
    function = 'drop'

    def initialize_options(self) -> None:
        self.type = None
        self.dataset = None
        self.predictor = None

    def finalize_options(self) -> None:
        if self.type:
            if self.type.lower() == 'd':
                self.function = 'drop'
            elif self.type.lower() == 'n':
                self.function = 'noise'
        if self.dataset:
            if self.dataset.lower() == 'b':
                self.datasets = [BreastCancer]
            elif self.dataset.lower() == 's':
                self.datasets = [SpliceJunction]
            elif self.dataset.lower() == 'c':
                self.datasets = [CensusIncome]
        if self.predictor:
            if self.predictor.lower() == 'u':
                self.predictor_names = ['uneducated']
            elif self.predictor.lower() == 'kins':
                self.predictor_names = ['kins']
            elif self.predictor.lower() == 'kill':
                self.predictor_names = ['kill']
            elif self.predictor.lower() == 'kbann':
                self.predictor_names = ['kbann']

    def run(self) -> None:
        get_custom_objects().update(NetBuilder.custom_objects)
        set_seed(SEED)
        for dataset in self.datasets:
            print(f'Running experiments for {dataset.name} dataset')
            data = pd.read_csv(dataset.file_name, header=0, sep=",", encoding='utf8')
            loss = 'binary_crossentropy'
            if dataset.name == CensusIncome.name:
                uneducated = generate_neural_network_census_income(self.metrics)
            elif dataset.name == SpliceJunction.name:
                loss = 'categorical_crossentropy'
                uneducated = generate_neural_network_splice_junction(self.metrics)
            else:
                uneducated = generate_neural_network_breast_cancer(self.metrics)
            for name in self.predictor_names:
                if name == 'uneducated':
                    if self.function == 'drop':
                        experiment_with_data_drop(data, uneducated, dataset.name, name, self.population_size,
                                                  self.metrics, loss=loss)
                    else:
                        experiment_with_data_noise(data, uneducated, dataset.name, name, self.population_size,
                                                   self.metrics, sigma=1, loss=loss)
                else:
                    if name == 'kins':
                        feature_mapping = {k: v for v, k in enumerate(data.columns[:-1])}
                        injector = Injector.kins(uneducated, feature_mapping)
                        knowledge = TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae
                    elif name == 'kill':
                        feature_mapping = {k: v for v, k in enumerate(data.columns[:-1])}
                        class_mapping = dataset.class_mapping_short
                        injector = Injector.kill(uneducated, class_mapping, feature_mapping)
                        knowledge = TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae
                    elif name == 'kbann':
                        feature_mapping = {k: v for v, k in enumerate(data.columns[:-1])}
                        injector = Injector.kbann(uneducated, feature_mapping)
                        knowledge = TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae
                    for k in knowledge:
                        k.trainable = True
                    predictor = injector.inject(knowledge)
                    if self.function == 'drop':
                        experiment_with_data_drop(data, predictor, dataset.name, name, self.population_size,
                                                  self.metrics, loss=loss)
                    else:
                        experiment_with_data_noise(data, predictor, dataset.name, name, self.population_size,
                                                   self.metrics, sigma=1, loss=loss)


class RunExperimentsDivergence(RunExperiments):

    def run(self) -> None:
        get_custom_objects().update(NetBuilder.custom_objects)
        set_seed(SEED)
        for dataset in self.datasets:
            print(f'Running experiments for {dataset.name} dataset')
            data = pd.read_csv(dataset.file_name, header=0, sep=",", encoding='utf8')
            if self.function == 'drop':
                compute_divergence_over_experiments_experiment_with_data_drop(data, dataset.name, self.population_size)
            else:
                compute_divergence_over_experiments_with_data_noise(data, dataset.name, self.population_size, sigma=1)


class ComputeMetrics(distutils.cmd.Command):
    description = 'print robustness metric'
    user_options = [('type=', 't', 'type of experiment (d[rop], n[oise])')]
    exp_type = None
    experiments = None
    function = 'drop'

    def initialize_options(self) -> None:
        self.type = None

    def finalize_options(self) -> None:
        if self.type:
            if self.type.lower() == 'd':
                self.function = 'drop'
            elif self.type.lower() == 'n':
                self.function = 'noise'

    def run(self) -> None:
        from results import PATH as RESULT_PATH
        datasets = [BreastCancer, SpliceJunction, CensusIncome]
        metrics = ['accuracy']
        robustness = {}
        for dataset in datasets:
            for metric in metrics:
                robustness = compute_robustness(self.function, dataset, metric)
            result = pd.DataFrame([robustness])
            result.to_csv(RESULT_PATH / self.function / dataset.name / 'robustness.csv', index=False)


class GenerateKnowledgeConfusionMatrix(distutils.cmd.Command):
    description = 'generate comparative distribution curves'
    user_options = [('type=', 't', 'type of experiment (d[rop], n[oise])')]
    exp_type = None
    experiments = None

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        datasets = [BreastCancer, SpliceJunction, CensusIncome]  # , SpliceJunction, CensusIncome BreastCancer
        metrics = ['accuracy']
        for dataset in datasets:
            data = pd.read_csv(dataset.file_name, header=0, sep=",", encoding='utf8')
            knowledge = TuProlog.from_file(KNOWLEDGE_PATH / dataset.knowledge_file_name).formulae
            if dataset.name == BreastCancer.name:
                predictor = generate_neural_network_breast_cancer(metrics)
            elif dataset.name == SpliceJunction.name:
                predictor = generate_neural_network_splice_junction(metrics)
            else:  # elif dataset.name == CensusIncome.name:
                predictor = generate_neural_network_census_income(metrics)
            feature_mapping = {k: v for v, k in enumerate(data.columns[:-1])}
            fuzzifier = NetBuilder(predictor.input, feature_mapping)
            output = Concatenate(axis=1)(fuzzifier.visit(knowledge))
            predictor = Model(predictor.input, output)
            predictor.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            cm = compute_confusion_matrix(data, predictor, dataset.name).to_numpy()
            plot_cm(cm, list(dataset.class_mapping_short.keys()), dataset.name)


class GeneratePlots(distutils.cmd.Command):
    description = 'generate plots'
    user_options = [('type=', 't', 'type of experiment (d[rop], n[oise])')]
    exp_type = None
    experiments = None

    def initialize_options(self) -> None:
        self.type = None

    def finalize_options(self) -> None:
        if self.type:
            if self.type.lower() == 'd':
                self.exp_type = 'drop'
                self.experiments = 20
            elif self.type.lower() == 'n':
                self.exp_type = 'noise'
                self.experiments = 11

    def run(self) -> None:
        from figures import plot_accuracy_distributions
        from results.drop import PATH as DROP_RESULT_PATH
        from results.noise import PATH as NOISE_RESULT_PATH

        path = DROP_RESULT_PATH if self.exp_type == 'drop' else NOISE_RESULT_PATH
        predictor_names = ['uneducated', 'kins', 'kill', 'kbann']
        datasets = [BreastCancer, SpliceJunction, CensusIncome]
        metrics = ['accuracy', 'precision', 'recall']
        for dataset in datasets:
            print(f'Generating plots for {dataset.name} dataset')
            for predictor in predictor_names:
                results = []
                directory = path / dataset.name / predictor
                if os.path.exists(directory):
                    files = os.listdir(directory)
                    files = sorted(files, key=lambda x: int("".join([i for i in x if i.isdigit()])))
                    files = [directory / f for f in files if f.endswith('.csv')]
                    if len(files) > 0:
                        first_drop = DROP_RESULT_PATH / dataset.name / predictor / '1.csv'
                        if first_drop not in files:
                            files.insert(0, first_drop)
                        for file in files:
                            results.append(pd.read_csv(file, header=0, sep=",", encoding='utf8'))
                        for metric in metrics:
                            plot_accuracy_distributions(results, dataset, self.exp_type, 5, self.experiments, predictor, metric)


class GenerateComparisonPlots(distutils.cmd.Command):
    description = 'generate comparison plots'
    user_options = [('type=', 't', 'type of experiment (d[rop], n[oise])')]
    exp_type = None
    experiments = None

    def initialize_options(self) -> None:
        self.type = None

    def finalize_options(self) -> None:
        if self.type:
            if self.type.lower() == 'd':
                self.exp_type = 'drop'
                self.experiments = 20
            elif self.type.lower() == 'n':
                self.exp_type = 'noise'
                self.experiments = 11

    def run(self) -> None:
        from figures import plot_distributions_comparison
        from results.drop import PATH as DROP_RESULT_PATH
        from results.noise import PATH as NOISE_RESULT_PATH

        path = DROP_RESULT_PATH if self.exp_type == 'drop' else NOISE_RESULT_PATH
        educated_predictors = ['kins', 'kill', 'kbann']
        datasets = [BreastCancer, SpliceJunction, CensusIncome]
        metric = 'accuracy'
        for dataset in datasets:
            print(f'Generating comparison plots for {dataset.name} dataset')
            directory1 = path / dataset.name / 'uneducated'
            files1 = os.listdir(directory1)
            files1 = sorted(files1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
            files1 = [directory1 / f for f in files1 if f.endswith('.csv')]
            first_drop = DROP_RESULT_PATH / dataset.name / 'uneducated' / '1.csv'
            if first_drop not in files1:
                files1.insert(0, first_drop)
            for educated in educated_predictors:
                directory2 = path / dataset.name / educated
                if not os.path.exists(directory2):
                    continue
                files2 = os.listdir(directory2)
                files2 = sorted(files2, key=lambda x: int("".join([i for i in x if i.isdigit()])))
                files2 = [directory2 / f for f in files2 if f.endswith('.csv')]
                first_drop = DROP_RESULT_PATH / dataset.name / educated / '1.csv'
                if first_drop not in files2:
                    files2.insert(0, first_drop)
                results1, results2 = [], []
                if 0 < len(files1) == len(files2):
                    for file in files1:
                        results1.append(pd.read_csv(file, header=0, sep=",", encoding='utf8'))
                    for file in files2:
                        results2.append(pd.read_csv(file, header=0, sep=",", encoding='utf8'))
                    plot_distributions_comparison(results1, results2, dataset, self.exp_type, 5, self.experiments, 'uneducated', educated, metric)


class GenerateComparativeDistributionCurves(distutils.cmd.Command):
    description = 'generate comparative distribution curves'
    user_options = [('type=', 't', 'type of experiment (d[rop], n[oise])')]
    exp_type = None
    experiments = None

    def initialize_options(self) -> None:
        self.type = None

    def finalize_options(self) -> None:
        if self.type:
            if self.type.lower() == 'd':
                self.exp_type = 'drop'
                self.experiments = 20
            elif self.type.lower() == 'n':
                self.exp_type = 'noise'
                self.experiments = 11

    def run(self) -> None:
        from figures import plot_average_accuracy_curves
        from results.drop import PATH as DROP_RESULT_PATH
        from results.noise import PATH as NOISE_RESULT_PATH

        path = DROP_RESULT_PATH if self.exp_type == 'drop' else NOISE_RESULT_PATH
        educated_predictors = ['kins', 'kill', 'kbann']
        datasets = [BreastCancer]
        metric = 'accuracy'
        for dataset in datasets:
            print(f'Generating comparative distribution curves for {dataset.name} dataset')
            directory1 = path / dataset.name / 'uneducated'
            files1 = os.listdir(directory1)
            files1 = sorted(files1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
            files1 = [directory1 / f for f in files1 if f.endswith('.csv')]
            first_drop = DROP_RESULT_PATH / dataset.name / 'uneducated' / '1.csv'
            if first_drop not in files1:
                files1.insert(0, first_drop)
            paths = [path / dataset.name / educated for educated in educated_predictors]
            paths = [p for p in paths if os.path.exists(path)]
            files_groups = [os.listdir(path) for path in paths]
            experiments = []
            tmp = []
            for file in files1:
                tmp.append(pd.read_csv(directory1 / file, header=0, sep=",", encoding='utf8'))
            experiments.append(tmp)
            for p, files in zip(paths, files_groups):
                tmp = []
                files = sorted(files, key=lambda x: int("".join([i for i in x if i.isdigit()])))
                complete_files = [p / file for file in files if file.endswith('.csv')]
                first_drop = DROP_RESULT_PATH / Path(*p.parts[-2:]) / '1.csv'
                if first_drop not in complete_files:
                    complete_files.insert(0, first_drop)
                for file in complete_files:
                    tmp.append(pd.read_csv(file, header=0, sep=",", encoding='utf8'))
                experiments.append(tmp)
            plot_average_accuracy_curves(experiments, dataset, self.exp_type, 5, self.experiments, educated_predictors, metric)


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
        'run_experiments': RunExperiments,
        'run_divergence': RunExperimentsDivergence,
        'compute_robustness': ComputeMetrics,
        'generate_plots': GeneratePlots,
        'generate_comparison_plots': GenerateComparisonPlots,
        'generate_comparative_distribution_curves': GenerateComparativeDistributionCurves,
        'generate_knowledge_confusion_matrix': GenerateKnowledgeConfusionMatrix,
    },
)
