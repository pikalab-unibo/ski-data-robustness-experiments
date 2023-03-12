from pathlib import Path
import numpy as np
import pandas as pd
from psyki.logic import Formula, DefinitionFormula, Clause
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Model
from data import CensusIncome, BreastCancer, SpliceJunction
from experiments import SEED, TEST_RATIO

PATH = Path(__file__).parents[0]


def compute_confusion_matrix(data: pd.DataFrame, predictor: Model, dataset_name: str) -> pd.DataFrame:
    _, test = train_test_split(data, test_size=TEST_RATIO, random_state=SEED, stratify=data.iloc[:, -1])
    x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    if dataset_name in (CensusIncome.name, BreastCancer.name):
        predictor = DecisionTreeClassifier(max_depth=20, max_leaf_nodes=20)
        train, _ = train_test_split(data, random_state=0, train_size=0.5)
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
        predictor.fit(train_x, train_y)

    y_pred = predictor.predict(x_test)
    if dataset_name == SpliceJunction.name:
        y_pred = pd.DataFrame(y_pred)
        y_pred = y_pred.apply(lambda x: np.argmax(x), axis=1)
    confusion_matrix = pd.crosstab(list(y_test), list(y_pred))
    return confusion_matrix


def compute_confusion_matrix_with_knowledge(data: pd.DataFrame, knowledge: list[DefinitionFormula]) -> pd.DataFrame:

    def create_filters(c: Clause) -> dict[str, tuple[str, str]]:
        results = {}
        textual_clause = str(c)
        filters = textual_clause.split(',')
        for f in filters:
            if '<' in f:
                feature, value = f.split('<')
                results[feature] = ('<', value)
            elif '>' in f:
                feature, value = f.split('>')
                results[feature] = ('>', value)
            else:
                print('Error')
        return results

    y_pred = []
    _, test = train_test_split(data, test_size=TEST_RATIO, random_state=SEED, stratify=data.iloc[:, -1])
    x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    results = {}
    tmp_x = x_test.copy()
    for f in knowledge:
        filters = create_filters(f.rhs)
        label = int(str(f.lhs.args.last))
        tmp_r = tmp_x.copy()
        for filter in filters:
            feature = filter[0]
            value = float(filter[1][1])
            if filter[1][0] == '<':
                r = x_test.loc[x_test[feature] >= value]
            else:
                r = x_test.loc[x_test[feature] <= value]
            tmp_r.sub(r)
        for index in tmp_r.index:
            results[index] = label

    confusion_matrix = pd.crosstab(list(y_test), list(y_pred))
    return confusion_matrix
