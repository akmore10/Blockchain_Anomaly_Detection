from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    n_classes = 2  # MNIST has 10 classes
    n_features = 165  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_Anomaly_Data() -> Dataset:
    features = pd.read_csv("elliptic_bitcoin_dataset\elliptic_txs_features.csv",header=None)
    classes = pd.read_csv("elliptic_bitcoin_dataset\elliptic_txs_classes.csv")
    

    tx_features = ["tx_feat_"+str(i) for i in range(2,95)]
    agg_features = ["agg_feat_"+str(i) for i in range(1,73)]
    features.columns = ['txId' ,'time_step'] + tx_features + agg_features
    features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')    
    features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)

    data = features[(features['class']=='1') | (features['class']=='2')]
    X_Tx = data[tx_features + agg_features]
    y_Tx = data['class']

    y_Tx = y_Tx.apply(lambda x: 0 if x == '2' else 1 )

    X_train_Tx, X_test_Tx, y_train_Tx, y_test_Tx = train_test_split(X_Tx,y_Tx,test_size=0.3,random_state=15,shuffle=False)
    
    return (X_train_Tx, y_train_Tx), (X_test_Tx, y_test_Tx)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    return list(zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions)))
