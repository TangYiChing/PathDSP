"""
calculate accuracy, auc, auprcfor classification task
calculate rmse, r2, pcc for regression task
"""
import numpy as np
import pandas as pd
import scipy.stats as scistat
import sklearn.metrics as skmts


def eval_classifier_performance(df, method='per-drug'):
    """
    :param df: dataframe with headers=[drug, cell, resp, prediction]
    :param metho: string indicating per-drug or per-cell, default=per-drug
    :return eval_df
    """
    # determine index 
    if method == 'per-drug':
        idx_str = 'drug'
    elif method == 'per-cell':
        idx_str = 'cell'
    else:
        print('ERROR: {:} not supported. use default=per-drug'.format(method))
    # collect result
    record_list = []
    # cacluate accuracy, auc, auprc
    for idx in df[idx_str].unique():
        # get data
        idx_df = df.loc[df[idx_str]==idx]
        # calculate scores
        yvalid_arr = idx_df['resp']
        y_pred = idx_df['prediction']
        y_pred_probas = idx_df['prediction_probability']
        auprc = skmts.average_precision_score(yvalid_arr, y_pred_probas)
        fpr, tpr, thresholds = skmts.roc_curve(yvalid_arr, y_pred_probas)
        acc = skmts.accuracy_score(yvalid_arr, y_pred)
        auc = skmts.auc(fpr, tpr)
        #logloss = skmts.log_loss(yvalid_arr, y_pred)
        precision = skmts.precision_score(yvalid_arr, y_pred, average='micro')
        recall = skmts.recall_score(yvalid_arr, y_pred, average='micro')
        f1 = skmts.f1_score(y_pred, yvalid_arr)
        # append to result
        record_list.append( (idx, acc, auc, auprc, precision, recall, f1) )
    # convert list 2 dataframe
    result_df = pd.DataFrame.from_records(record_list, columns=[idx_str, 'acc', 'auc', 'auprc', 'precision', 'recall', 'f1'])
    print(result_df)
    
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
        
        

if __name__ == "__main__":
    #fname = 'test.LogisticRegression.prediction.txt' #'./example_output/GDSC.SMILE.TARGETnNEIGHBOR.binary-EXP.binary-MUTCNV.binary.RESP.binary.LogisticRegression.prediction.txt'
    #prediction_df = pd.read_csv(fname, header=0, sep="\t")
    #eval_classifier_performance(prediction_df, method='per-cell')

    fname = '/raid/ytang4/FNN_GDSC_CHEM_EXP_MUT_CNV/GDSC.CHEM-EXP-MUT-CNV.Seed42.FNN.cv_10.Prediction.txt'
    df = pd.read_csv(fname, header=0, index_col=[0,1], sep="\t")
    eval_regressor_performance(df, 'resp', 'prediction')

