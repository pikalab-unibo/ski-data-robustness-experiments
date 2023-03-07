import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.python.keras.utils.np_utils import to_categorical
from data import BreastCancer, SpliceJunction, CensusIncome
from results.drop import PATH as DROP_RESULTS_PATH
from results.noise import PATH as NOISE_RESULTS_PATH
from statistics import apply_noise

TEST_RATIO = 1 / 3
DROP_RATIO = 0.05  # 5% of the data is dropped
N_STEP_DROP = 19  # from 0 to N_STEPS, 0 means no noise, N_STEPS means maximum noise = DROP_RATIO * N_STEPS = 95%
N_STEP_NOISE = 10
NOISE = 1
EPOCHS = 100
BATCH_SIZE = 32
VERBOSE = 0
SEED = 0


def _generate_uneducated_neural_network(data: pd.DataFrame, neurons_per_layers: list[int], metrics: list,
                                        activation: str = 'relu', optimizer: str = 'adam',
                                        loss: str = 'categorical_crossentropy') -> Model:
    input_layer = Input(shape=(data.shape[1] - 1,))
    x = Dense(neurons_per_layers[0], activation=activation)(input_layer)
    for i in range(len(neurons_per_layers) - 1):
        x = Dense(neurons_per_layers[i], activation=activation)(x)
    output = neurons_per_layers[-1]
    if output == 1:
        x = Dense(output, activation='sigmoid')(x)
    else:
        x = Dense(output, activation='softmax')(x)
    model = Model(input_layer, x)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def generate_neural_network_breast_cancer(metrics: list) -> Model:
    data = pd.read_csv(BreastCancer.file_name, sep=",", header=0, encoding='utf8')
    return _generate_uneducated_neural_network(data, [16, 8, 2], metrics)


def generate_neural_network_splice_junction(metrics: list) -> Model:
    data = pd.read_csv(SpliceJunction.file_name, sep=",", header=0, encoding='utf8')
    return _generate_uneducated_neural_network(data, [64, 32, 3], metrics)


def generate_neural_network_census_income(metrics: list) -> Model:
    data = pd.read_csv(CensusIncome.file_name, sep=",", header=0, encoding='utf8')
    return _generate_uneducated_neural_network(data, [32, 16, 2], metrics)


def _create_missing_directories(path: Path, data_name: str, ski_name: str):
    if not os.path.exists(path / data_name):
        os.mkdir(path / data_name)
    if not os.path.exists(path / (data_name + os.sep + ski_name)):
        os.mkdir(path / (data_name + os.sep + ski_name))


def train_and_cumulate_results(train: pd.DataFrame, predictor: Model, ski_name: str, metrics: list, loss: str, results: pd.DataFrame, x_test, y_test, p: int):
    x_train = train.iloc[:, :-1]
    y_train = to_categorical(train.iloc[:, -1:])
    set_seed(p)
    if ski_name == 'kill':
        predictor_copy = predictor.copy()
    else:
        predictor_copy = clone_model(predictor)
    predictor_copy.set_weights(predictor.get_weights())
    predictor_copy.compile(loss=loss, optimizer='adam', metrics=metrics)
    predictor_copy.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
    if ski_name == 'kill':
        predictor_copy = predictor_copy.remove_constraints()
        predictor_copy.compile(loss=loss, optimizer='adam', metrics=metrics)
    evaluation = predictor_copy.evaluate(x_test, y_test, verbose=VERBOSE)
    results.loc[p] = evaluation[1:]
    print("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f}".format(evaluation[1], evaluation[2], evaluation[3]))


def experiment_with_data_drop(data: pd.DataFrame, predictor: Model, data_name: str, ski_name: str, population: int,
                              metrics: list, drop_size: float = DROP_RATIO, n_steps: int = N_STEP_DROP,
                              test_size: float = TEST_RATIO, seed: int = SEED, loss: str = 'categorical_crossentropy'):
    print("Experiment with data drop: {} - {}".format(data_name, ski_name))
    n_steps += 1  # Because the first step is the original dataset
    train, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data.iloc[:, -1])
    x_test = test.iloc[:, :-1]
    y_test = to_categorical(test.iloc[:, -1:])
    if ski_name == 'kbann':
        set_seed(seed)
        predictor = clone_model(predictor)
    for i in range(n_steps):
        print("\n\nStep {}/{}\n".format(i + 1, n_steps))
        _create_missing_directories(DROP_RESULTS_PATH, data_name, ski_name)
        file_name = DROP_RESULTS_PATH / (data_name + os.sep + ski_name + os.sep + "{}.csv".format(i + 1))
        if os.path.exists(file_name):
            print("File {} already exists".format(file_name))
            continue
        else:
            results = pd.DataFrame(columns=['accuracy', 'precision', 'recall'])
            f = drop_size * i
            for p in range(population):
                print("Population {}/{}".format(p + 1, population))
                if f > 0:
                    new_train, _ = train_test_split(train, test_size=f, random_state=seed + p, stratify=train.iloc[:, -1])
                else:
                    new_train = train
                train_and_cumulate_results(new_train, predictor, ski_name, metrics, loss, results, x_test, y_test, p)
            results.to_csv(file_name, index=False)


def experiment_with_data_noise(data: pd.DataFrame, predictor: Model, data_name: str, ski_name: str, population: int,
                               metrics: list, mu: float = 0, sigma: float = 0, n_steps: int = N_STEP_NOISE,
                               noise: float = NOISE, test_size: float = TEST_RATIO, seed: int = SEED,
                               loss: str = 'categorical_crossentropy'):
    print("Experiment with data noise: {} - {}".format(data_name, ski_name))
    train, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data.iloc[:, -1])
    x_test = test.iloc[:, :-1]
    y_test = to_categorical(test.iloc[:, -1:])
    if ski_name == 'kbann':
        set_seed(seed)
        predictor = clone_model(predictor)
    for i in range(n_steps):
        print("\n\nStep {}/{}\n".format(i + 1, n_steps))
        _create_missing_directories(NOISE_RESULTS_PATH, data_name, ski_name)
        file_name = NOISE_RESULTS_PATH / (data_name + os.sep + ski_name + os.sep + "{}.csv".format(i + 1))
        if os.path.exists(file_name):
            print("File {} already exists".format(file_name))
            continue
        else:
            results = pd.DataFrame(columns=['accuracy', 'precision', 'recall'])
            for p in range(population):
                print("Population {}/{}".format(p + 1, population))
                new_train = train.copy()
                # For now we only support noise on numerical columns
                column_data_types = {k: 'int' for k in new_train.columns[:-1]}
                new_train = apply_noise(new_train, mu, sigma + (i + 1) * noise, column_data_types, seed + p + i)
                train_and_cumulate_results(new_train, predictor, ski_name, metrics, loss, results, x_test, y_test, p)
            results.to_csv(file_name, index=False)
