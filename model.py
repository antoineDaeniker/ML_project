#%%
import datetime
from utils.proj1_helpers import *
from utils.utils import *
import time

def run_model():
    DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here
    y, X, Xt, ids = load_csv_data(DATA_TRAIN_PATH)
    print(y.shape, X.shape)

    DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here
    y_test, X_test, Xt_test, ids_test = load_csv_data(DATA_TEST_PATH)


    #%%
    ##################### DATA TRAIN PROCESSING #####################
    data_irr, irr_ind = delete_irr_features(X, 0.5)
    data_irr_corr, corr_ind,_ = feature_correlation(data_irr, 0.9)
    data_irr_corr_norm, norm_ind = normalize_data(data_irr_corr)

    rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
    rmv_idx = np.insert(rmv_idx, -1, norm_ind)
    rmv_idx = np.unique(rmv_idx)
    print("DATA TRAIN PROCESSING DONE")
    ###########################################################

    ##################### TRAINING ############################
    # Define the parameters of the algorithm.
    max_iters = 800
    gamma = 0.1
    batch_size = 1

    # Initialization
    w_initial = np.zeros(data_irr_corr_norm.shape[1])

    # Start SGD.
    start_time = datetime.datetime.now()
    sgd_losses, sgd_ws = gradient_descent(
        y, data_irr_corr_norm, w_initial, max_iters, gamma)

    # Save weights
    np.savetxt('sgd_model.csv', np.asarray(sgd_ws), delimiter=',')

    # Load weights
    sgd_ws = np.loadtxt('sgd_model.csv', delimiter=',')

    for i in rmv_idx:
        sgd_ws = np.insert(sgd_ws, i, 0)
    print("\nRESULTING W : ", sgd_ws)

    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
    print("TRAINING DONE")
    ###########################################################

    ################# DATA TEST PROCESSING ####################
    #data_test = np.delete(X_test, rmv_idx, axis=1)
    print("DATA TEST PROCESSING DONE")
    ###########################################################

    ######################## ACCURACY #########################
    predictions = get_predictions(sgd_ws, X_test, y_test, y_test.shape[0])

    create_csv_submission(ids_test, predictions, f'data/submissions/submission_{time.strftime("%Y%m%d-%H%M%S")}.csv')
    # %%

    def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
        num_row = y.shape[0]
        interval = int(num_row / k_fold)
        np.random.seed(seed)
        indices = np.random.permutation(num_row)
        k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
        return np.array(k_indices)
    
    def cross_validation(y, tx, k_fold, seed=1, k, lambda_, initial_w, max_iters, gamma):
    """return the loss of ridge regression."""
        rmse_tr = []
        rmse_te = []
        k_indices = build_k_indices(y, k_fold, seed)
    # ***************************************************
    # get k'th subgroup in test, others in train
    # ***************************************************
        for k in range(k_fold):

            te_indice = k_indices[k]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indice = tr_indice.reshape(-1)
            y_te = y[te_indice]
            y_tr = y[tr_indice]
            tx_te = x[te_indice]
            tx_tr = x[tr_indice]
    
            w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_ , initial_w, max_iters, gamma)
        
            loss_tr = np.sqrt(2 * calculate_loss(y_tr, x_tr, w))
            loss_te = np.sqrt(2 * calculate_loss(y_te, x_te, w))

            rmse_tr.append(loss_tr)
            rmse_te.append(loss_te)
        return rmse_tr, rmse_te, w