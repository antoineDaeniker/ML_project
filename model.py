# %%
from datetime import datetime

from implementations import reg_logistic_regression
from utils.preprocessing_utils import make_prediction_split_for_submission, preprocess_train_data, preprocess_train_data_split
from utils.io_utils import *
from utils.implementation_utils import *
import time

import logging

logger = logging.getLogger(__name__)


def train(X, y, rmv_idx, max_iters=800, gamma=1e-7, batch_size=1, save_weights=False, add_bias_term=True):
    #w, loss = ridge_regression(y, X, lambda_=1e-8)
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

    return w, loss


def run_model_split(save_weights=False, retrain=True, internal_test=True, create_submission=False, add_bias_term=True):
    y, X, Xt, ids = load_csv_data('data/train.csv')
    print('Data shape: ', y.shape, X.shape)
    degrees = [3, 4, 5, 6, 7, 8]
    losses = []
    for degree in degrees:
        X_list, y_list, rmv_idx_list = preprocess_train_data_split(X, y, degree)
        ws = []
        losses_split = []
        for i, (y, X, rmv_idx), in enumerate(zip(y_list, X_list, rmv_idx_list)):
            if add_bias_term:
                X = np.concatenate((np.ones(X.shape[0])[:, np.newaxis], X), axis=1)
            k_fold = 10
            k_indices = build_k_indices(y, k_fold)
            w_split = []
            losses_k = []
            for k in range(k_fold):
                start_time = datetime.now()
                X_train, y_train, X_test, y_test = split_cross_validation(y, X, k_indices, k)
                y_train_dist = np.asarray((np.unique(y_train, return_counts=True))).T
                y_test_dist = np.asarray((np.unique(y_test, return_counts=True))).T
                #with np.printoptions(precision=0, suppress=True):
                #    print(f'y_train distribution: {y_train_dist} \ny_test distribution: {y_test_dist}')

                if not retrain:
                    w = np.loadtxt('sgd_model.csv', delimiter=',')
                else:
                    w, loss = train(X_train, y_train, rmv_idx)
                    losses_k.append(loss)
                    w_split.append(w)
                    end_time = datetime.now()
                    exection_time = (end_time - start_time).total_seconds()
                    print("Model training time={t:.3f} seconds".format(t=exection_time))

                if internal_test:
                    print(f'Test for datasplit : {i} and k {k}')
                    test(w, X_test, y_test)
            ws.append(w_split)
            losses_split.append(np.mean(losses_k))
        losses.append(np.mean(losses_split))

        ws_best = find_best_w(ws, losses)
        print('Best weights : ',ws_best)

    if save_weights:
        ws_best_array = np.array(ws_best[0], ws_best[1], ws_best[2], ws_best[3])
        #np.savetxt('sgd_model_split.csv', np.asarray(ws_best_array), delimiter=',')
    if create_submission:
        print('Creating submission')
        y_te, X_te, _, ids_te = load_csv_data('data/test.csv')
        new_y_pred = make_prediction_split_for_submission(y_te, X_te, ids_te, rmv_idx_list, ws_best)
        create_csv_submission(ids_te, new_y_pred, f'data/submissions/submission_split{time.strftime("%Y%m%d-%H%M%S")}.csv')

def test(w, X_test, y_test):
    y_pred = predict_labels(w, X_test)
    y_test[np.where(y_test <= 0)] = -1
    accuracy = get_accuracy(y_pred, y_test)
    print(f'Model accuracy: {accuracy}')


def run_model(save_weights=False, retrain=True, internal_test=True, create_submission=False, custom_split=True):
    if custom_split:
        run_model_split()
    else :
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
            #for i in rmv_idx:
            #    w_ = np.insert(w_, i, 0)
            _, X_test, _, ids_test = load_csv_data('data/test.csv')
            new_X_test = data_for_test_submit(X_test, rmv_idx)
            print('X_test Process Shape : ', new_X_test.shape)
            y_pred = predict_labels(w_, new_X_test)
            create_csv_submission(ids_test, y_pred, f'data/submissions/submission_{time.strftime("%Y%m%d-%H%M%S")}.csv')


"""
    CROSS VALIDATION
    NORMALISATION without removed features
    BIAS TERM WITH WEIGHT 1 (after normalising) ##DONE
    SPLITTING DATA                              Still need to do prediction and do one submit
    
    TODO final submission with all data
    TODO polynomial 
"""