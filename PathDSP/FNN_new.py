"""
Train a neural network for regression task:
    cv: 10
    batch size: 8
    initializer: He normal initializer
    optimizer: AdamMax
    learning rate: 0.0004
    loss: RMSE

Calculate RMSE at once, Oct. 3, 2020 revised
"""


import argparse
import numpy as np
import pandas as pd
import scipy.stats as scistat
from datetime import datetime

import sklearn.preprocessing as skpre
import sklearn.model_selection as skms
import sklearn.metrics as skmts
import sklearn.utils as skut

import torch as tch
import torch.utils.data as tchud

import myModel as mynet
import myFit as myfit
import myDataloader as mydl
import myDatasplit as mysplit
import myUtility as myutil
#import myPlotter as myplot
import myMetrics as mymts
import polars as pl

#import shap as sp

class RMSELoss(tch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = tch.nn.MSELoss()
        loss = tch.sqrt(criterion(x, y) + eps)
        return loss


def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - ss_res / ss_tot
    return r2

def cal_time(end, start):
    '''return time spent'''
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start),datetimeFormat)
    return spend



def fit(net, train_dl, valid_dl, epochs, learning_rate, device, opt_fn):
    """
    Return train and valid performance including loss

    :param net: model
    :param train_dl: train dataloader
    :param valid_dl: valid dataloader
    :param epochs: integer representing EPOCH
    :param learning_rate: float representing LEARNING_RATE
    :param device: string representing cpu or cuda:0
    :param opt_fn: optimization function in torch (e.g., tch.optim.Adam)
    :param loss_fn: loss function in torch (e.g., tch.nn.MSELoss)
    """
    # setup
    criterion = RMSELoss() # setup LOSS function
    optimizer = opt_fn(net.parameters(), lr=learning_rate, weight_decay=1e-5) # setup optimizer
    net = net.to(device) # load the network onto the device
    trainloss_list = [] # metrics: MSE, size equals to EPOCH
    validloss_list = [] # metrics: MSE, size equals to EPOCH
    validr2_list = [] # metrics: r2, size equals to EPOCH
    early_stopping = myutil.EarlyStopping(patience=30, verbose=True) # initialize the early_stopping
    # repeat the training for EPOCH times
    start_total = datetime.now()
    for epoch in range(epochs):
        ## training phase
        start = datetime.now()
        net.train()
        # initial loss
        train_epoch_loss = 0.0 # save loss for each epoch, batch by batch
        for i, (X_train, y_train) in enumerate(train_dl):
            X_train, y_train = X_train.to(device), y_train.to(device) # load data onto the device
            y_train_pred  = net(X_train) # train result
            train_loss = criterion(y_train_pred, y_train.float()) # calculate loss
            optimizer.zero_grad() # clear gradients
            train_loss.backward() # backpropagation
            #### add this if you have gradient explosion problem ###
            clip_value = 5
            tch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            ########climp gradient within -5 ~ 5 ###################
            optimizer.step() # update weights
            train_epoch_loss += train_loss.item() # adding loss from each batch
        # calculate total loss of all batches
        avg_train_loss = train_epoch_loss / len(train_dl)
        trainloss_list.append( avg_train_loss )
        print('epoch ' + str(epoch) + ' :[Finished in {:}]'.format(cal_time(datetime.now(), start)))
        ## validation phase
        with tch.no_grad():
            net.eval()
            valid_epoch_loss = 0.0 # save loss for each epoch, batch by batch
            ss_res = 0.0
            ss_tot = 0.0
            for i, (X_valid, y_valid) in enumerate(valid_dl):
                X_valid, y_valid = X_valid.to(device), y_valid.to(device) # load data onto the device
                y_valid_pred  = net(X_valid) # valid result
                valid_loss = criterion(y_valid_pred, y_valid.float())#y_valid.unsqueeze(1)) # calculate loss
                valid_epoch_loss += valid_loss.item() # adding loss from each batch
                ss_res += tch.sum((y_valid_pred - y_valid.float())**2)
                ss_tot += tch.sum((y_valid_pred - y_valid.mean())**2)
        
        # calculate total loss of all batches, and append to result list
        avg_valid_loss = valid_epoch_loss / len(valid_dl)
        validloss_list.append( avg_valid_loss)
        valid_r2 = 1 - ss_res / ss_tot
        validr2_list.append(valid_r2.cpu().numpy())
        # display print message
        #print('epoch={:}/{:}, train loss={:.5f}, valid loss={:.5f}'.format(
        #       epoch+1, epochs, train_epoch_loss / len(train_dl),
        #                        valid_epoch_loss / len(valid_dl)))
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss, net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print('Total time (all epochs) :[Finished in {:}]'.format(cal_time(datetime.now(), start_total)))
    # load the last checkpoint with the best model
    net.load_state_dict(tch.load('checkpoint.pt'))

    return  net, trainloss_list, validloss_list, validr2_list

def predict(net, test_dl, device):
    """
    Return prediction list

    :param net: model
    :param train_dl: train dataloader
    :param device: string representing cpu or cuda:0
    """
    # create result lists
    prediction_list = list()

    with tch.no_grad():
        net = net.to(device) # load the network onto the device
        net.eval()
        for i, (X_test, y_test) in enumerate(test_dl):
            X_test, y_test = X_test.to(device), y_test.to(device) # load data onto the device
            y_test_pred  = net(X_test) # test result
            # bring data back to cpu in np.array format, and append to result lists
            prediction_list.append( y_test_pred.cpu().numpy() )
            #print(prediction_list)

    # merge all batches
    prediction_list  = np.vstack(prediction_list)
    prediction_list = np.hstack(prediction_list).tolist()
    # return
    return prediction_list


def main(params):
    start_time = datetime.now()
    # load data
    print('loadinig data')
    # train_df = pd.read_csv(params['train_data'], header=0, index_col=[0,1], sep="\t")
    # val_df = pd.read_csv(params['val_data'], header=0, index_col=[0,1], sep="\t")
    # test_df = pd.read_csv(params['test_data'], header=0, index_col=[0,1], sep="\t")
    train_df = pl.read_csv(params['train_data'], separator = "\t").to_pandas()
    val_df = pl.read_csv(params['val_data'], separator = "\t").to_pandas()
    
    # shuffle
    #sdf = skut.shuffle(df, random_state=params["seed_int"])

    # set parameters
    myutil.set_seed(params["seed_int"])
    device = myutil.get_device(uth=params["gpu_int"])
    #kFold = params["cv_int"]
    learning_rate = params['learning_rate']
    epoch = params['epochs']
    batch_size = params['batch_size']
    opt_fn = tch.optim.Adam

    # create result list
    # loss_df_list = []
    # score_df_list = []
    # ytest_df_list = []
    # shap_df_list = []
    # # train with cross-validation
    #kf = skms.KFold(n_splits=kFold, random_state=params['seed_int'], shuffle=True)
    #X_df = train_df.iloc[:, 0:-1]
    #y_df = train_df.iloc[:, -1]
    # save best model with lowest RMSE
#     best_rmse = 10000
#     best_model = None
#     best_fold = 0
# #    for i, (train_index, test_index) in enumerate(kf.split(X_df, y_df)):
    #n_fold = i+1
    #print('Fold={:}/{:}'.format(n_fold, params['cv_int']))
    # get train/test splits
    Xtrain_arr = train_df.iloc[:, 0:-1].values
    Xvalid_arr = val_df.iloc[:, 0:-1].values
    ytrain_arr = train_df.iloc[:, -1].values
    yvalid_arr = val_df.iloc[:, -1].values    
    
    # get train/valid splits from train
    #Xtrain_arr, Xvalid_arr, ytrain_arr, yvalid_arr = skms.train_test_split(Xtrain_arr, ytrain_arr,
    #                                                                        test_size=0.1, random_state=params['seed_int'])
    #print('    train={:}, valid={:}, test={:}'.format(Xtrain_arr.shape, Xvalid_arr.shape, Xtest_arr.shape))
    # prepare dataframe for output
    #ytest_df = test_df.iloc[:, -1].to_frame()
    # convert to numpy array
    Xtrain_arr = np.array(Xtrain_arr).astype('float32')
    Xvalid_arr = np.array(Xvalid_arr).astype('float32')
    ytrain_arr = np.array(ytrain_arr).astype('float32')
    yvalid_arr = np.array(yvalid_arr).astype('float32')
    # create mini-batch
    train_dataset = mydl.NumpyDataset(tch.from_numpy(Xtrain_arr), tch.from_numpy(ytrain_arr))
    valid_dataset = mydl.NumpyDataset(tch.from_numpy(Xvalid_arr), tch.from_numpy(yvalid_arr))
    train_dl = tchud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = tchud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # initial weight
    def init_weights(m):
        if type(m) == tch.nn.Linear:
            tch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    # load model
    n_features = Xtrain_arr.shape[1]
    net = mynet.FNN(n_features)
    net.apply(init_weights)
    # fit data with model
    print('start training process')
    trained_net, train_loss_list, valid_loss_list, valid_r2_list = fit(net, train_dl, valid_dl, epoch, learning_rate, device, opt_fn)
    # if rmse <= best_rmse:
    #     best_rmse = rmse
    #     best_fold = n_fold
    #     best_model = trained_net
    #     print('best model so far at fold={:}, rmse={:}'.format(best_fold, best_rmse))
    
    
    # if params['shap_bool'] == True:
    #     print('calculate shapely values')
    #     # random select 100 samples as baseline
    #     train_dataset = mydl.NumpyDataset(tch.from_numpy(Xtrain_arr), tch.from_numpy(ytrain_arr))
    #     train_dl = tchud.DataLoader(train_dataset, batch_size=200, shuffle=True)
    #     background, lbl  = next(iter(train_dl))
    #     explainer = sp.DeepExplainer(trained_net, background[:100].to(device))
    #     shap_arr = explainer.shap_values(tch.from_numpy(Xtest_arr))
    #     shap_df = pd.DataFrame(shap_arr, index=ytest_df.index, columns=X_df.columns)
    #     # append to result
    #     shap_df_list.append(shap_df)
        
    # collect result
    loss_df = pd.DataFrame({'epoch':[i+1 for i in range(len(train_loss_list))],
                            'train loss':train_loss_list, 
                            'valid loss': valid_loss_list,
                            'valid r2': valid_r2_list})

    #loss_df_list.append(loss_df)
    #ytest_df_list.append(ytest_df)
    # end of fold
    #trained_net = None

    # save to output
    #all_ytest_df = pd.concat(ytest_df_list, axis=0)
    #all_loss_df = pd.concat(loss_df_list, axis=0)
    loss_df.to_csv(params['data_dir'] + '/Loss.txt', header=True, index=False, sep="\t")
    # if params['shap_bool'] == True:
    #     all_shap_df = pd.concat(shap_df_list, axis=0)
    #     all_shap_df.to_csv(params['output'] + '.FNN.cv_' + str(params['cv_int']) + '.SHAP.txt', header=True, index=True, sep="\t")

    # make train/valid loss plots
    best_model = trained_net
    tch.save(best_model.state_dict(), params['data_dir'] + '/model.pt')
    print( '[Finished in {:}]'.format(myutil.cal_time(datetime.now(), start_time)) )
    # display evaluation metrics of all folds
    #mse, rmse, r_square, pccy = mymts.eval_regressor_performance(all_ytest_df, 'response', 'prediction')




if __name__ == "__main__":
    main()