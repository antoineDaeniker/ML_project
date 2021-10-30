import numpy as np
import logging

from utils.implementation_utils import build_poly, delete_irr_features, feature_correlation, normalize_data, split_data_for_test_submit, subdivide_data
from utils.io_utils import predict_labels

logger = logging.getLogger(__name__)

def update_labels(y):
    """ Change y labels from y E {-1, 1} to y E {0, 1} """
    y[np.where(y == -1)] = 0
    return y


def preprocess_train_data(X, y):
    logger.info('Preprocessing data')

    _, irr_ind = delete_irr_features(X, 0.5)
    _, corr_ind,_ = feature_correlation(X, 0.9)
    _, norm_ind = normalize_data(X)

    rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
    rmv_idx = np.insert(rmv_idx, -1, norm_ind)
    rmv_idx = np.unique(rmv_idx)
    print(rmv_idx)
    data_reduce = np.delete(X, rmv_idx, axis=1)
    data_irr_corr_norm,_ = normalize_data(data_reduce)

    y = update_labels(y)

    return data_irr_corr_norm, y, rmv_idx


def preprocess_train_data_split(X, y, update_label=False):
    logger.info('Preprocessing data')
    X_list, y_list, feat_ind = subdivide_data(X, y)
    data_split = []
    y_split = []
    rmv_idx_split = []

    for X, y in zip(X_list, y_list):
        _, irr_ind = delete_irr_features(X, 0.5)
        _, corr_ind,_ = feature_correlation(X, 0.9)
        _, norm_ind = normalize_data(X)

        rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
        rmv_idx = np.insert(rmv_idx, -1, norm_ind)
        rmv_idx = np.insert(rmv_idx, -1, feat_ind)
        rmv_idx = np.unique(rmv_idx)
        print('Removed features indexes : ', rmv_idx)
        data_reduce = np.delete(X, rmv_idx, axis=1)
        data_poly = build_poly(data_reduce, 8)
        data_irr_corr_norm,_ = normalize_data(data_poly)

        if update_label:
            y = update_labels(y)

        data_split.append(data_irr_corr_norm)
        y_split.append(y)
        rmv_idx_split.append(rmv_idx)

    return data_split, y_split, rmv_idx_split


def make_prediction_split_for_submission(y_te, X_te, ids_te, rmv_idx_list, ws_best):
    y_pred_list = []
    X_test_list, y_test_list, ids_list = split_data_for_test_submit(ids_te, X_te, y_te, rmv_idx_list)
    for ws, rmv_idx, sub_X_test in zip(ws_best, rmv_idx_list, X_test_list):
        #for idx in rmv_idx:
        #    ws = np.insert(ws, idx, 0)
        #sub_X_test, _ = normalize_data(sub_X_test)
        sub_X_test = np.concatenate((np.ones(sub_X_test.shape[0])[:, np.newaxis], sub_X_test), axis=1)
        y_pred = predict_labels(ws, sub_X_test)
        y_pred_list.append(y_pred)
    y_pred_conc = np.concatenate((y_pred_list[0], y_pred_list[1], y_pred_list[2] ,y_pred_list[3]), axis=0)[:, np.newaxis]
    ids_conc = np.concatenate((ids_list[0], ids_list[1], ids_list[2] ,ids_list[3]), axis=0)[:, np.newaxis]
    ids_y_conc = np.concatenate((ids_conc, y_pred_conc), axis=1)
    sorted_y_conc = ids_y_conc[ids_y_conc[:, 0].argsort()]
    new_y_pred = np.squeeze(np.delete(sorted_y_conc, 0, axis=1))

    return new_y_pred

