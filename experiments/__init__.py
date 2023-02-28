import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.framework.random_seed import set_seed
from data import BreastCancer, SpliceJunction, CensusIncome
from results.drop import PATH as DROP_RESULTS_PATH


TEST_RATIO = 0.2
DROP_RATIO = 0.05  # 5% of the data is dropped
N_STEPS = 15
EPOCHS = 100
BATCH_SIZE = 32
VERBOSE = 0
SEED = 0


def _generate_uneducated_neural_network(data: pd.DataFrame, neurons_per_layers: list[int], metrics: list,
                                        activation: str = 'relu', optimizer: str = 'adam',
                                        loss: str = 'binary_crossentropy') -> Model:
    input_layer = Input(shape=(data.shape[1] - 1,))
    x = Dense(neurons_per_layers[0], activation=activation)(input_layer)
    for i in range(len(neurons_per_layers) - 1):
        x = Dense(neurons_per_layers[i], activation=activation)(x)
    x = Dense(neurons_per_layers[-1], activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def generate_neural_network_breast_cancer(metrics: list) -> Model:
    data = pd.read_csv(BreastCancer.file_name, sep=",", header=0, encoding='utf8')
    return _generate_uneducated_neural_network(data, [16, 8, 1], metrics)


def generate_neural_network_splice_junction(metrics: list) -> Model:
    data = pd.read_csv(SpliceJunction.file_name, sep=",", header=0, encoding='utf8')
    return _generate_uneducated_neural_network(data, [64, 32, 3], metrics)


def generate_neural_network_census_income(metrics: list) -> Model:
    data = pd.read_csv(CensusIncome.file_name, sep=",", header=0, encoding='utf8')
    return _generate_uneducated_neural_network(data, [32, 16, 2], metrics)


def experiment_with_data_drop(data: pd.DataFrame, predictor: Model, data_name: str, ski_name: str, population: int,
                              metrics: list, drop_size: float = DROP_RATIO, n_steps: int = N_STEPS,
                              test_size: float = TEST_RATIO, seed: int = SEED):
    print("Experiment with data drop: {} - {}".format(data_name, ski_name))
    set_seed(seed)
    train, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data.iloc[:, -1])
    for i in range(n_steps):
        print("\n\nStep {}/{}\n".format(i + 1, n_steps))
        results = pd.DataFrame(columns=['accuracy', 'precision', 'recall'])
        f = drop_size * i
        for p in range(population):
            print("Population {}/{}".format(p + 1, population))
            if f > 0:
                new_train, _ = train_test_split(train, test_size=f, random_state=seed + p, stratify=train.iloc[:, -1])
            else:
                new_train = train
            predictor_copy = clone_model(predictor)
            predictor_copy.set_weights(predictor.get_weights())
            predictor_copy.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
            predictor_copy.fit(new_train.iloc[:, :-1], new_train.iloc[:, -1:], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
            evaluation = predictor_copy.evaluate(test.iloc[:, :-1], test.iloc[:, -1:], verbose=VERBOSE)
            results.loc[p] = evaluation[1:]
            print("Accuracy: {:.2f} - Precision: {:.2f} - Recall: {:.2f}".format(evaluation[1], evaluation[2], evaluation[3]))
        results.to_csv(DROP_RESULTS_PATH / "{}_{}_{}.csv".format(data_name, ski_name, i + 1), index=False)
