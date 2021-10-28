# %%
from datetime import datetime

from implementations import reg_logistic_regression
from utils.preprocessing_utils import preprocess_train_data
from utils.io_utils import *
from utils.implementation_utils import *
import time

import logging

logger = logging.getLogger(__name__)


def train(X, y, rmv_idx, max_iters=800, gamma=1e-5, batch_size=1, save_weights=False):
    w, loss = reg_logistic_regression(
        y=y,
        tx=X,
        lambda_=1e-8,
        initial_w=np.zeros(X.shape[1]),
        max_iters=max_iters,
        gamma=gamma,
    )

    logger.info(f'Final loss value for trained model: {loss}')

    # for i in rmv_idx:
    #     w = np.insert(w, i, 0)

    return w


def test(w, X_test, y_test):
    y_pred = predict_labels(w, X_test)
    y_test[np.where(y_test <= 0)] = -1
    accuracy = get_accuracy(y_pred, y_test)
    print(f'Model accuracy: {accuracy}')


def run_model(save_weights=True, retrain=True, internal_test=True, create_submission=True):
    y, X, Xt, ids = load_csv_data('data/train.csv')
    print('Data shape: ', y.shape, X.shape)
    X, y, rmv_idx = preprocess_train_data(X, y)
    X_train, y_train, X_test, y_test = split_data(X, y, 0.6)
    y_train_dist = np.asarray((np.unique(y_train, return_counts=True))).T
    y_test_dist = np.asarray((np.unique(y_test, return_counts=True))).T
    with np.printoptions(precision=0, suppress=True):
        print(f'y_train distribution: {y_train_dist} \ny_test distribution: {y_test_dist}')

    if not retrain:
        w = np.loadtxt('sgd_model.csv', delimiter=',')
    else:
        start_time = datetime.now()
        w = train(X_train, y_train, rmv_idx)
        if save_weights:
            np.savetxt('sgd_model.csv', np.asarray(w), delimiter=',')
        end_time = datetime.now()
        exection_time = (end_time - start_time).total_seconds()
        print("Model training time={t:.3f} seconds".format(t=exection_time))

    if internal_test:
        test(w, X_test, y_test)

    if create_submission:
        print('Creating submission')
        w_ = w
        for i in rmv_idx:
            w_ = np.insert(w_, i, 0)
        _, X_test, _, ids_test = load_csv_data('data/test.csv')
        y_pred = predict_labels(w_, X_test)
        create_csv_submission(ids_test, y_pred, f'data/submissions/submission_{time.strftime("%Y%m%d-%H%M%S")}.csv')


"""
    CROSS VALIDATION
    NORMALISATION without removed features
    BIAS TERM WITH WEIGHT 1 (after normalising)
    SPLITTING DATA 
"""