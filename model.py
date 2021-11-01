# %%
from collections import defaultdict
from datetime import datetime

from implementations import reg_logistic_regression
from utils.preprocessing_utils import make_prediction_split_for_submission, preprocess_train_data, \
    preprocess_train_data_split
from utils.io_utils import *
from utils.implementation_utils import *
import time
import json

import logging

logger = logging.getLogger(__name__)


def create_run_config(method=reg_logistic_regression, max_iters=2000, lambda_=1e-8, start_degree=-3, end_degree=8,
                      include_half=True, include_cross_terms=True):
    return dict(method=method, max_iters=max_iters, lambda_=lambda_, start_degree=start_degree, end_degree=end_degree,
                include_half=include_half, include_cross_terms=include_cross_terms)


def get_run_configs(k=10):
    """
        each element specifies method name, lambda, polynomial start degree,
        polynomial end degree, include half, include cross terms
    """
    configs = [
                  create_run_config(lambda_=1e-7, start_degree=-2, end_degree=2),
                  create_run_config(lambda_=1e-6, start_degree=-2, end_degree=2),
                  create_run_config(lambda_=1e-5, start_degree=-2, end_degree=2),
        
                  create_run_config(lambda_=1e-7, start_degree=-3, end_degree=3),
                  create_run_config(lambda_=1e-6, start_degree=-3, end_degree=3),
                  create_run_config(lambda_=1e-5, start_degree=-3, end_degree=3),
        
                  create_run_config(lambda_=1e-7, start_degree=-4, end_degree=4),
                  create_run_config(lambda_=1e-6, start_degree=-4, end_degree=4),
                  create_run_config(lambda_=1e-5, start_degree=-4, end_degree=4),
        
                  create_run_config(lambda_=1e-6, start_degree=-5, end_degree=5),
                  # create_run_config(max_iters=2000, lambda_=1e-6),
                  # create_run_config(method=ridge_regression, lambda_=1e-7),
                  # create_run_config(method=ridge_regression, lambda_=1e-6),
                  # create_run_config(method=ridge_regression, lambda_=1e-5),
                  # create_run_config(method=ridge_regression, lambda_=1e-4),
              ][:k]

    assert len(configs) == k

    return configs


def train(X, y, rmv_idx, method=reg_logistic_regression, max_iters=800, gamma=1e-7, lambda_=1e-8, batch_size=1,
          start_degree=-3, end_degree=8, include_half=True, include_cross_terms=True):

    if method == reg_logistic_regression:
        w, loss = reg_logistic_regression(
            y=y,
            tx=X,
            lambda_=lambda_,
            initial_w=np.zeros(X.shape[1]),
            max_iters=max_iters,
            gamma=gamma,
        )
    elif method == ridge_regression:
        w, loss = ridge_regression(y, X, lambda_=1e-8)
    else:
        raise ValueError('Please specify valid method name')

    logger.info(f'Final loss value for trained model: {loss}')

    # for i in rmv_idx:
    #     w = np.insert(w, i, 0)

    return w, loss


def run_model_split(save_weights=False, retrain=True, internal_test=True, create_submission=False, add_bias_term=True,
                    apply_cross_validation=True):
    y, X, Xt, ids = load_csv_data('data/train.csv')
    print('Data shape: ', y.shape, X.shape)
    X_list, y_list, rmv_idx_list = preprocess_train_data_split(X, y)  # doesn't do any train / test splitting
    ws = []
    losses = []
    lambda_poly_plots = []
    for i, (y, X, rmv_idx), in enumerate(zip(y_list, X_list, rmv_idx_list)):
        lambda_poly_plot = []
        if add_bias_term:
            X = np.concatenate((np.ones(X.shape[0])[:, np.newaxis], X), axis=1)
        w_split = []
        losses_split = []
        if apply_cross_validation:
            k_fold = 10
            k_indices = build_k_indices(y, k_fold)
        else:
            k_fold = 1
        for k in range(k_fold):
            start_time = datetime.now()
            current_config = get_run_configs(k=k_fold)[k]
            print(f'Training with config: {current_config}')
            if apply_cross_validation:
                X_train, y_train, X_test, y_test = split_cross_validation(y, X, k_indices, k, training_config=current_config)
                y_train_dist = np.asarray((np.unique(y_train, return_counts=True))).T
                y_test_dist = np.asarray((np.unique(y_test, return_counts=True))).T
                # with np.printoptions(precision=0, suppress=True):
                #    print(f'y_train distribution: {y_train_dist} \ny_test distribution: {y_test_dist}')
            else:
                X_train, y_train = X, y

            if not retrain:
                w = np.loadtxt('sgd_model.csv', delimiter=',')
            else:
                if apply_cross_validation:
                    w, loss = train(X_train, y_train, rmv_idx, **current_config)
                else:
                    w, loss = train(X_train, y_train, rmv_idx)
                losses_split.append(loss)
                w_split.append(w)
                end_time = datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("Model training time={t:.3f} seconds".format(t=exection_time))

            if internal_test and apply_cross_validation:
                print(f'Test for datasplit : {i} and k {k}')
                accuracy = test(w, X_test, y_test)
                current_config['method'] = current_config['method'].__name__
                lambda_poly_plot.append((current_config, accuracy))
        ws.append(w_split)
        losses.append(losses_split)
        lambda_poly_plots.append(lambda_poly_plot)

    ws_best = find_best_w(ws, losses)
    print('Best weights : ', ws_best)

    try:
        with open(f'data/lambda_poly_plots_data{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f_out:
            json.dump(lambda_poly_plots, f_out)
    except Exception as e:
        print('Write error: ', e)

    if save_weights:
        ws_best_array = np.array(ws_best[0], ws_best[1], ws_best[2], ws_best[3])
        np.savetxt('sgd_model_split.csv', np.asarray(ws_best_array), delimiter=',')
    if create_submission:
        print('Creating submission')
        y_te, X_te, _, ids_te = load_csv_data('data/test.csv')
        new_y_pred = make_prediction_split_for_submission(y_te, X_te, ids_te, rmv_idx_list, ws_best)
        create_csv_submission(ids_te, new_y_pred,
                              f'data/submissions/submission_split{time.strftime("%Y%m%d-%H%M%S")}.csv')


def test(w, X_test, y_test):
    y_pred = predict_labels(w, X_test)
    y_test[np.where(y_test <= 0)] = -1
    accuracy = get_accuracy(y_pred, y_test)
    print(f'Model accuracy: {accuracy}')
    return accuracy


def run_model(save_weights=True, retrain=True, internal_test=True, create_submission=False, custom_split=True):
    if custom_split:
        run_model_split()
    else:
        y, X, Xt, ids = load_csv_data('data/train.csv')
        print('Data shape: ', y.shape, X.shape)
        X, y, rmv_idx = preprocess_train_data(X, y)
        print('X Process Shape : ', X.shape)
        k_fold = 10
        k_indices = build_k_indices(y, k_fold)
        ws = []
        losses = []
        for k in range(k_fold):
            X_train, y_train, X_test, y_test = split_cross_validation(y, X, k_indices, k)
            y_train_dist = np.asarray((np.unique(y_train, return_counts=True))).T
            y_test_dist = np.asarray((np.unique(y_test, return_counts=True))).T
            with np.printoptions(precision=0, suppress=True):
                print(f'y_train distribution: {y_train_dist} \ny_test distribution: {y_test_dist}')

            if not retrain:
                w = np.loadtxt('sgd_model.csv', delimiter=',')
            else:
                start_time = datetime.now()
                w, loss = train(X_train, y_train, rmv_idx)
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
    BIAS TERM WITH WEIGHT 1 (after normalising) ##DONE
    SPLITTING DATA                              Still need to do prediction and do one submit
    
    TODO final submission with all data
    TODO polynomial 
"""
