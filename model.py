# %%
from collections import defaultdict
from datetime import datetime

from implementations import least_squares, least_squares_SGD, reg_logistic_regression, ridge_regression
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
    """
    methode: the method to be use for the model
    max_iters: the number of iteration to do to train the model
    lambda_: regularize term
    start_degree: the starting degree for polynomial expension feature
    end_degree: the ending degree for polynomial expension feature
    include_half: boolean specifies whether to include square rooted feature terms in polynomial
    include_cross_terms: boolean specifies whether to include bivariate feature terms in polynomial (e.g. X1X2, X1X3)

    return: summerize of all parameters we use for the model
    """
    return dict(method=method, max_iters=max_iters, lambda_=lambda_, start_degree=start_degree, end_degree=end_degree,
                include_half=include_half, include_cross_terms=include_cross_terms)


def get_run_configs(k=10):
    """
        each element specifies method name, lambda, polynomial start degree,
        polynomial end degree, include half, include cross terms
    """
    configs = [
                  create_run_config(method=least_squares, start_degree=1, end_degree=1, include_half=False, include_cross_terms=False),
                  create_run_config(method=ridge_regression, max_iters=800),
                  create_run_config(method=least_squares_SGD),
                  create_run_config(lambda_=1e-7, start_degree=-2, end_degree=2),
                  create_run_config(lambda_=1e-6, start_degree=-2, end_degree=2),
                  create_run_config(lambda_=1e-5, start_degree=-2, end_degree=2),
        
                  create_run_config(lambda_=1e-7, start_degree=-3, end_degree=3),
                  create_run_config(lambda_=1e-6, start_degree=-3, end_degree=3),
                  create_run_config(lambda_=1e-5, start_degree=-3, end_degree=3),
        
                  create_run_config(lambda_=1e-7, start_degree=-4, end_degree=4),
                  #create_run_config(lambda_=1e-6, start_degree=-4, end_degree=4),
                  #create_run_config(lambda_=1e-5, start_degree=-4, end_degree=4),
        
                  #create_run_config(lambda_=1e-6, start_degree=-5, end_degree=5),
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
    """
    Training procedure

    X: 2-D numpy array of data samples
    y: 1-D numpy array of labels
    methode: the method to be use for the model
    max_iters: int, the number of iteration to do to train the model
    gamma: float, step use gradient descent
    lambda_: float, regularize term
    batch_size: int, the number of sample we take at each iter for stochastic gradient
    start_degree: int, the starting degree for polynomial expension feature
    end_degree: int, the ending degree for polynomial expension feature
    include_half: bool, specifies whether to include square rooted feature terms in polynomial
    include_cross_terms: bool, specifies whether to include bivariate feature terms in polynomial (e.g. X1X2, X1X3)

    return: w: 1-D numpy array, parameters output of the model to compute predictions
            loss: float, loss compute using the w parameter
    """

    initial_w = np.zeros(X.shape[1])
    if method == ridge_regression:
        w, loss = ridge_regression(
            y=y,
            tx=X,
            lambda_=lambda_
        )
    elif method == least_squares:
        w, loss = least_squares(
            y=y, 
            tx=X
        )
    elif method == reg_logistic_regression:
        w, loss = reg_logistic_regression(
            y=y,
            tx=X,
            lambda_=lambda_,
            initial_w=initial_w,
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

def test(w, X_test, y_test):
    """
    w: 1-D numpy array of parameter result from the training procedure
    X_test: 2-D numpy array of test sample from the validation set
    y_test: 1-D numpy array of labels from the validation set
    returns: accuracy of the predictions
    """

    y_pred = predict_labels(w, X_test)
    y_test[np.where(y_test <= 0)] = -1
    accuracy = get_accuracy(y_pred, y_test)
    print(f'Model accuracy: {accuracy}')
    return accuracy
    
def run_model_split(save_weights=False, retrain=True, internal_test=True, create_submission=True, add_bias_term=True,
                    apply_cross_validation=True):
    """
    Model using split data set

    save_weights: bool, True if we want to save the w paremeter, False if we don't want them to be save
    retrain: bool, True if we want to generate new w for the model, False if we reuse previously computed w
    internal_test: bool, True if we want to compute the accuracy of the validation set, False if not
    create_submission: bool, True if we want to generate csv file to be submit, False if not
    add_bias_term: bool, True if we want to add bias term, False if not
    apply_cross_validation: bool, True if we split the data set into train and validation set
    """

    y, X, Xt, ids = load_csv_data('data/train.csv')
    training_config = create_run_config(method=reg_logistic_regression, lambda_=1e-8, start_degree=-8, end_degree=8, max_iters=800)
    print('Data shape: ', y.shape, X.shape)
    X_list, y_list, rmv_idx_list = preprocess_train_data_split(X, y, training_config=training_config)  # doesn't do any train / test splitting
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
        for k in range(1):
            start_time = datetime.now()
            current_config = get_run_configs(k=k_fold)[0] if apply_cross_validation else training_config
            print(f'Training with config: {current_config}')
            if apply_cross_validation:
                X_train, y_train, X_test, y_test = split_cross_validation(y, X, k_indices, k, training_config=None)
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
                    w, loss = train(X_train, y_train, rmv_idx, **training_config)
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
        new_y_pred = make_prediction_split_for_submission(y_te, X_te, ids_te, rmv_idx_list, ws_best, training_config=training_config)
        create_csv_submission(ids_te, new_y_pred,
                              f'data/submissions/submission_split{time.strftime("%Y%m%d-%H%M%S")}.csv')







"""
    CROSS VALIDATION
    NORMALISATION without removed features
    BIAS TERM WITH WEIGHT 1 (after normalising) ##DONE
    SPLITTING DATA                              Still need to do prediction and do one submit
    
    TODO final submission with all data
    TODO polynomial 
"""
