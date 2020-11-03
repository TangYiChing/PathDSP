"""
Return
1. Average RMSE, R2, PCC of 10-fold cross validation on test set (outter loop)
2. Average Feature Importance of 5-fold cv hyperparameter optimization on train set (inner loop)

Use Bayesian Optimization to find the best parameters
for xgboost regressor
    1. learning_rate: (0.01, 0.1, 0.5, 1.0)
    2. max_depth: (1,5,10,15,20)
    3. reg_lambda: (0, 1.0, 10.0)
    4. gamma: (0.0, 0.25, 1.0)
"""


# built-in pkgs
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import scipy.stats as scistat
from datetime import datetime
import memory_profiler
import gc
# xgboost, shap
import catboost as cab
import xgboost as xgb
import shap as sp# to obtain feature importance by shapley values

# sklearn
import sklearn.utils as skut
import sklearn.metrics as skmts
import sklearn.model_selection as skms
import sklearn.model_selection as skms
import sklearn.svm as sksvm
import sklearn.ensemble as sken
import sklearn.linear_model as sklm

# Bayesian Optimization
from skopt import BayesSearchCV
import skopt.space as skosp
import joblib as jlb # to save trained model

def parse_parameter():
    parser = argparse.ArgumentParser(description = "Nested Cross Validation of XGBoost Regressor")
    parser.add_argument("-i", "--input_path",
                        required = True,
                        help = "path to input file with feature and label")
    parser.add_argument("-s", "--seed_int",
                        required = False,
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-outerK", "--outerK_int",
                        required = False,
                        default = 5,
                        type = int,
                        help = "K fold cross validation for estimate generalization performance of a model. default=5")
    parser.add_argument("-innerK", "--innerK_int",
                        required = False,
                        default = 5,
                        type = int,
                        help = "K fold cross validation for hyperparameter tuning. default=5")
    parser.add_argument("-n", "--njob_int",
                        required = False,
                        type = int,
                        default = 1,
                        help = "number of jobs to run in parallel for hyperparameer tuning process")
    parser.add_argument("-t", "--test_path",
                        required = False,
                        default = None,
                        help = "use independent test set for evaluation if given")
    parser.add_argument("-m", "--model_str",
                        required = True,
                        choices = ['ElasticNet', 'SVM', 'RandomForest', 'XGBoost', 'CatBoost'])
    parser.add_argument("-shap", "--shap_bool",
                        required = False,
                        type = bool,
                        default = False,
                        help = "calculate feature importance by shapley values if True, default=False")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "path to output files")
    return parser.parse_args()


def cal_time(end, start):
    """return time spent"""
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start),datetimeFormat)
    return spend

def tuning_hyperparameters(model, param_dict, cv_int, X_arr, y_arr, seed_int, fit_param_dict=None):
    """return optimal parameters"""
    m1 = memory_profiler.memory_usage()
    t1 = datetime.now()
    if fit_param_dict == None:
        optimizer = BayesSearchCV(estimator = model,
                             search_spaces = param_dict,
                             scoring = 'neg_root_mean_squared_error', # RMSE
                             n_jobs = 1, # BE AWARE: the total number of cores used will be njob_int*cv_int, DO NOT CHANGE THIS
                             cv = cv_int,
                             refit = True, # to refit a final model with the entire training set
                             random_state = seed_int,
                             verbose = False)
    else:
        print('    fit_param={:}'.format(fit_param_dict))
        optimizer = BayesSearchCV(estimator = model,
                             search_spaces = param_dict,
                             fit_params = fit_param_dict, # parameters for fit method
                             scoring = 'neg_root_mean_squared_error', # RMSE
                             n_jobs = 1, # BE AWARE: the total number of cores used will be njob_int*cv_int, DO NOT CHANGE THIS
                             cv = cv_int,
                             refit = True, # to refit a final model with the entire training set
                             random_state = seed_int,
                             verbose = False)
    # return
    cv_results = optimizer.fit(X_arr, y_arr)
    #optimal_param_dict = cv_results.best_params_
    m2 = memory_profiler.memory_usage()
    t2 = datetime.now()
    print("It took {:} Secs and {:} Mb to execute this method=tuning_hyperparameters".format(cal_time(t2, t1), (m2[0] - m1[0])))
    return cv_results

def fit(model, Xtrain_arr, ytrain_arr, Xtest_arr, ytest_arr, metric_str="rmse", early_stop_int=10):
    """return prediction"""
    eval_set = [(Xtest_arr, ytest_arr)]
    model.fit(Xtrain_arr, ytrain_arr, early_stopping_rounds=int(early_stop_int), eval_metric=metric_str, eval_set=eval_set, verbose=False)
    y_pred = model.predict(Xtest_arr)
    yield y_pred


if __name__ == "__main__":
    # timer 
    start = datetime.now()

    # get args
    args = parse_parameter()


    # settings
    np.random.seed(args.seed_int)
    model_choice_dict = {'ElasticNet': sklm.ElasticNet(max_iter=1000, fit_intercept=True, random_state=args.seed_int),
                         'RandomForest': sken.RandomForestRegressor(n_jobs=args.njob_int, random_state=args.seed_int),
                         'SVM': sksvm.SVR(max_iter=1000),
                         'CatBoost': cab.CatBoostRegressor(eval_metric='RMSE', iterations=1000, boosting_type= 'Plain', thread_count=args.njob_int, random_state=args.seed_int, silent=True),
                         'XGBoost': xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000, tree_method='hist', nthread=args.njob_int, random_state=args.seed_int, verbose=True)}

    model_param_dict = {'ElasticNet': {'l1_ratio': (0.20, 0.40, 0.60, 0.80), 'alpha': (0.001, 0.01, 1, 10)},
                        'RandomForest': {'max_depth': (3, 4, 5)},
                        'SVM': {'kernel': ("rbf", "linear"), 'C': (0.01, 1, 10)},
                        'CatBoost': {'max_depth': (3, 4, 5), 'learning_rate': (0.01, 0.1, 0.5, 1), 'l2_leaf_reg': (1, 5, 7)},
                        'XGBoost': {'max_depth': (3, 4, 5), 'learning_rate': (0.01, 0.1, 0.5, 1), 'gamma': (0.0, 0.25, 1.0), 'reg_lambda': (0.0, 1.0, 10.0)}}


    # load data
    print(datetime.now(), 'loading inputs')
    df = pd.read_csv(args.input_path, header=0, index_col=[0,1], sep="\t")

    # shuffle
    print(datetime.now(), 'shuffling data')
    sdf = skut.shuffle(df, random_state=args.seed_int)

    # split data
    print(datetime.now(), 'subsetting X, y')
    X_df = sdf.iloc[:, 0:-1]
    y_df = sdf.iloc[:, -1]

    # conver to numpy array
    X_arr = X_df.values.astype(np.float64)
    y_arr = y_df.values.astype(np.float64)

    # result list
    idx_list = [] # collect test_idx
    pred_list = [] # collect prediction for all folds
    shap_list = [] # collect shapley values for all folds (test set)

    # outer K fold
    best_rmse = 100 # defined by test RMSE
    best_param_dict = {} # defined by test RMSE
    best_model_object = None # defined by test RMSE
    print(datetime.now(), 'split outer folds')
    kf = skms.KFold(n_splits=args.outerK_int, shuffle=True, random_state=args.seed_int) 
    for i, (train_idx, test_idx) in enumerate(kf.split(X_df)):
        n_fold = i + 1
        m1 = memory_profiler.memory_usage()[0]
        # get train/test
        Xtrain_arr, Xtest_arr = X_arr[train_idx], X_arr[test_idx]
        ytrain_arr, ytest_arr = y_arr[train_idx], y_arr[test_idx]
        Xtrain_arr, Xvalid_arr, ytrain_arr, yvalid_arr = skms.train_test_split(Xtrain_arr, ytrain_arr,
                                                                               test_size=0.1, random_state=args.seed_int)

        print('    Xtrain={:} ytrain={:} | Xvalid={:} yvalid={:} | Xtest={:}, ytest={:}'.format(
                   Xtrain_arr.shape, ytrain_arr.shape, Xvalid_arr.shape, yvalid_arr.shape, Xtest_arr.shape, ytest_arr.shape))

        # hyperparameter tuning on train
        print('    fold={:}/{:}'.format(n_fold, args.outerK_int))
        model = model_choice_dict[args.model_str]
        if args.model_str == 'XGBoost':
            fit_params =  {'early_stopping_rounds': 10,
                           'verbose':False,
                           'eval_set':[(Xvalid_arr, yvalid_arr)],
                           'eval_metric': "rmse"}
            cv_results = tuning_hyperparameters(model, model_param_dict[args.model_str], args.innerK_int, Xtrain_arr, ytrain_arr, args.seed_int, fit_param_dict=fit_params)
        elif args.model_str == 'CatBoost':
            fit_params =  {'early_stopping_rounds': 10,
                           'verbose':False,
                           'eval_set':[(Xvalid_arr, yvalid_arr)]}
            cv_results = tuning_hyperparameters(model, model_param_dict[args.model_str], args.innerK_int, Xtrain_arr, ytrain_arr, args.seed_int, fit_param_dict=fit_params)

        else:    
            cv_results = tuning_hyperparameters(model, model_param_dict[args.model_str], args.innerK_int, Xtrain_arr, ytrain_arr, args.seed_int, fit_param_dict=None)
        optimal_model = cv_results.best_estimator_
        optimal_params = cv_results.best_params_
        #print('    best parameters={:}'.format(best_params))
        
        # evaluate on the hold out dataset
        print('    evaluate on the hold out set')
        y_pred = optimal_model.predict(Xtest_arr)
        mse = skmts.mean_squared_error(ytest_arr, y_pred)
        rmse = np.sqrt(mse)
        r_square = skmts.r2_score(ytest_arr, y_pred)
        pcc, pval = scistat.pearsonr(ytest_arr, y_pred)

        # calculate feature importance by shapley values
        if args.shap_bool == True and args.model_str in ['XGBoost', 'CatBoost', 'RandomForest']:
            print('    calculating SHAP values on test set')
            explainer = sp.TreeExplainer(optimal_model)
            shap_arr = explainer.shap_values(Xtest_arr)
            shap_list.append(shap_arr)

        # append prediction to result list
        pred_list.append(y_pred)
        idx_list.append(test_idx)


        # save best performing model for later use
        if rmse <= best_rmse:
            best_rmse = rmse
            #best_param_dict = optimal_params
            best_model_object = optimal_model
            jlb.dump(best_model_object, args.output_path+".best_model.dat")
            print('    best RMSE for far at fold={:}, RMSE={:.5f}'.format(n_fold, rmse))
            print('    best params={:}'.format(optimal_params))

        # end of outerCV
        model = None
        optimal_model = None 
        cv_results ={}
        del model
        del optimal_model
        del cv_results
        gc.collect()
        m2 = memory_profiler.memory_usage()[0]
        print('end fold, memory usage={:}, total={:}'.format((m2-m1), memory_profiler.memory_usage()[0]))

    # merge prediction of all folds
    print(datetime.now(), 'collecting results of all folds')
    index = y_df.iloc[np.concatenate(idx_list, 0)].to_frame().index
    pred_df = pd.DataFrame(np.concatenate(pred_list, 0), columns=['prediction'], index=index)
    pred_df = pd.concat([y_df.to_frame(), pred_df], axis=1)
    pred_df.to_csv(args.output_path + '.Prediction.txt', header=True, index=True, sep="\t")
    if len(shap_list) > 2:
        shap_df = pd.DataFrame(np.concatenate(shap_list, 0), columns=X_df.columns, index=index)
        shap_df.to_csv(args.output_path+'.SHAP.txt', header=True, index=True, sep="\t")

    # evaluation metrics for all folds
    print(datetime.now(), 'obtrain final performance')
    mse = skmts.mean_squared_error(pred_df['resp'], pred_df['prediction'])
    rmse = np.sqrt(mse)
    r_square = skmts.r2_score(pred_df['resp'], pred_df['prediction'])
    pcc, pval = scistat.pearsonr(pred_df['resp'], pred_df['prediction'])
    print(datetime.now(), 'obtain performance on all folds')
    print('    RMSE={:.5f}, R2={:.5f}, PCC={:.5f}'.format(rmse, r_square, pcc))
    print('[Finished in {:}]'.format(cal_time(datetime.now(), start)))

    # validation on independent test set
    if args.test_path != None:
        print(datetime.now(), 'testing on indepedent set')
        Xtrain_arr, Xvalid_arr, ytrain_arr, yvalid_arr = skms.train_test_split(X_arr, y_arr, test_size=0.2, random_state=args.seed_int)
        # fit model on whole train data
        print(datetime.now(), 'fitting whole data with the best performing model on train data')
        eval_set  = [(Xvalid_arr, yvalid_arr)]
        if args.model_str == 'XGBoost':
            best_model_object.fit(X_arr, y_arr, early_stopping_rounds=10, eval_set=eval_set, eval_metric="rmse", verbose=True)
        elif args.model_str == 'CatBoost':
            best_model_object.fit(X_arr, y_arr, early_stopping_rounds=10, eval_set=eval_set,  verbose=True)
        else:
            best_model_object.fit(X_arr, y_arr)

        # test the model on independent set
        print(datetime.now(), 'validation on the test data')
        ind_df = pd.read_csv(args.test_path, header=0, index_col=[0,1], sep="\t")
        # split X (feature) and y (label)
        ind_X_df = ind_df.iloc[:, 0:-1]
        ind_y_df = ind_df.iloc[:, -1]
        # conver to numpy array
        ind_X_arr = ind_X_df.values.astype(np.float32)
        ind_y_arr = ind_y_df.values.astype(np.float32)
        y_pred = best_model_object.predict(ind_X_arr)
        # evaluation metrics
        mse = skmts.mean_squared_error(ind_y_arr, y_pred)
        rmse = np.sqrt(mse)
        r_square = skmts.r2_score(ind_y_arr, y_pred)
        pcc, pval = scistat.pearsonr(ind_y_arr, y_pred)
        ind_y_df = ind_y_df.to_frame()
        ind_y_df.loc[:, 'prediction'] = list(y_pred)
        ind_y_df.to_csv(args.output_path + '.IndepedentSet.Prediction.txt', header=True, index=True, sep="\t")
        print('    test on independent set: RMSE={:}, R2={:}, PCC={:}'.format(rmse, r_square, pcc))
