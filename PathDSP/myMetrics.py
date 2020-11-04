"""
calculate rmse, r2, pcc for regression task
"""
import numpy as np
import pandas as pd
import scipy.stats as scistat
import sklearn.metrics as skmts
    
def eval_regressor_performance(df, y_true_col, y_pred_col):
    """
    return RMSE, R2, PCC
    """
    # get data
    ytest_arr = df[y_true_col]
    prediction_list = df[y_pred_col]

    # evaluation metrics
    mse = skmts.mean_squared_error(ytest_arr, prediction_list)
    rmse = np.sqrt(mse)
    r_square = skmts.r2_score(ytest_arr, prediction_list)
    pcc, pval = scistat.pearsonr(ytest_arr, prediction_list)
    print('MSE={:}, RMSE={:}, R2={:}, PCC={:}'.format(
           mse, rmse, r_square, pcc))
    return mse, rmse, r_square, pcc
