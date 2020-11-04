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
import myPlotter as myplot
import myMetrics as mymts

import shap as sp

class RMSELoss(tch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = tch.nn.MSELoss()
        loss = tch.sqrt(criterion(x, y) + eps)
        return loss

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
    early_stopping = myutil.EarlyStopping(patience=30, verbose=True) # initialize the early_stopping
    # repeat the training for EPOCH times
    for epoch in range(epochs):
        ## training phase
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
        ## validation phase
        with tch.no_grad():
            net.eval()
            valid_epoch_loss = 0.0 # save loss for each epoch, batch by batch
            for i, (X_valid, y_valid) in enumerate(valid_dl):
                X_valid, y_valid = X_valid.to(device), y_valid.to(device) # load data onto the device
                y_valid_pred  = net(X_valid) # valid result
                valid_loss = criterion(y_valid_pred, y_valid.float())#y_valid.unsqueeze(1)) # calculate loss
                valid_epoch_loss += valid_loss.item() # adding loss from each batch
        # calculate total loss of all batches, and append to result list
        avg_valid_loss = valid_epoch_loss / len(valid_dl)
        validloss_list.append( avg_valid_loss)

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
        
    # load the last checkpoint with the best model
    net.load_state_dict(tch.load('checkpoint.pt'))

    return  net, trainloss_list, validloss_list

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

# define arguments
def parse_parameter():
    parser = argparse.ArgumentParser(description = "Train a feedforward")
    parser.add_argument("-i", "--input_path",
                        required = True,
                        help = "input path")
    parser.add_argument("-s", "--seed_int",
                        required = False,
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-c", "--cv_int",
                        required = False,
                        default = 10,
                        type = int,
                        help = "K fold cross validation. default=10")
    parser.add_argument("-g", "--gpu_int",
                        default = 0,
                        type = int,
                        help = "assign the n-th GPU")
    parser.add_argument("-shap", "--shap_bool",
                        default = False,
                        type = bool,
                        help = "enable SHAP if True")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "output path")
    return parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.now()
    # get args
    args = parse_parameter()

    # load data
    df = pd.read_csv(args.input_path, header=0, index_col=[0,1], sep="\t")

    # shuffle
    sdf = skut.shuffle(df, random_state=args.seed_int)

    # set parameters
    myutil.set_seed(args.seed_int)
    device = myutil.get_device(uth=args.gpu_int)
    kFold = args.cv_int
    learning_rate = 0.0004
    epoch = 800
    batch_size = 12
    opt_fn = tch.optim.Adam

    # create result list
    loss_df_list = []
    score_df_list = []
    ytest_df_list = []
    shap_df_list = []
    # train with cross-validation
    kf = skms.KFold(n_splits=kFold, random_state=args.seed_int, shuffle=True)
    X_df = sdf.iloc[:, 0:-1]
    y_df = sdf.iloc[:, -1]
    # save best model with lowest RMSE
    best_rmse = 10000
    best_model = None
    best_fold = 0
    for i, (train_index, test_index) in enumerate(kf.split(X_df, y_df)):
        n_fold = i+1
        print('Fold={:}/{:}'.format(n_fold, args.cv_int))
        # get train/test splits
        Xtrain_arr = X_df.values[train_index]
        Xtest_arr = X_df.values[test_index]
        ytrain_arr = y_df.values[train_index]
        ytest_arr = y_df.values[test_index]
        # get train/valid splits from train
        Xtrain_arr, Xvalid_arr, ytrain_arr, yvalid_arr = skms.train_test_split(Xtrain_arr, ytrain_arr,
                                                                               test_size=0.1, random_state=args.seed_int)
        print('    train={:}, valid={:}, test={:}'.format(Xtrain_arr.shape, Xvalid_arr.shape, Xtest_arr.shape))
        # prepare dataframe for output
        ytest_df = y_df.iloc[test_index].to_frame()
        # convert to numpy array
        Xtrain_arr = np.array(Xtrain_arr).astype('float32')
        Xvalid_arr = np.array(Xvalid_arr).astype('float32')
        Xtest_arr = np.array(Xtest_arr).astype('float32')
        ytrain_arr = np.array(ytrain_arr).astype('float32')
        yvalid_arr = np.array(yvalid_arr).astype('float32')
        ytest_arr = np.array(ytest_arr).astype('float32')
        # create mini-batch
        train_dataset = mydl.NumpyDataset(tch.from_numpy(Xtrain_arr), tch.from_numpy(ytrain_arr))
        valid_dataset = mydl.NumpyDataset(tch.from_numpy(Xvalid_arr), tch.from_numpy(yvalid_arr))
        test_dataset = mydl.NumpyDataset(tch.from_numpy(Xtest_arr), tch.from_numpy(ytest_arr))
        train_dl = tchud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dl = tchud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dl = tchud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
        trained_net, train_loss_list, valid_loss_list = fit(net, train_dl, valid_dl, epoch, learning_rate, device, opt_fn)
        prediction_list = predict(trained_net, test_dl, device)
        # evaluation metrics
        mse = skmts.mean_squared_error(ytest_arr, prediction_list)
        rmse = np.sqrt(mse)
        if rmse <= best_rmse:
            best_rmse = rmse
            best_fold = n_fold
            best_model = trained_net
            print('best model so far at fold={:}, rmse={:}'.format(best_fold, best_rmse))
        
     
        if args.shap_bool == True:
            print('calculate shapely values')
            # random select 100 samples as baseline
            train_dataset = mydl.NumpyDataset(tch.from_numpy(Xtrain_arr), tch.from_numpy(ytrain_arr))
            train_dl = tchud.DataLoader(train_dataset, batch_size=200, shuffle=True)
            background, lbl  = next(iter(train_dl))
            explainer = sp.DeepExplainer(trained_net, background[:100].to(device))
            shap_arr = explainer.shap_values(tch.from_numpy(Xtest_arr))
            shap_df = pd.DataFrame(shap_arr, index=ytest_df.index, columns=X_df.columns)
            # append to result
            shap_df_list.append(shap_df)
            
        # collect result
        loss_df = pd.DataFrame({'fold':[n_fold]*len(train_loss_list), 
                                'epoch':[i+1 for i in range(len(train_loss_list))],
                                'train loss':train_loss_list, 
                                'valid loss': valid_loss_list})
        ytest_df['prediction'] = prediction_list
        ytest_df['fold'] = n_fold
        loss_df_list.append(loss_df)
        ytest_df_list.append(ytest_df)
        # end of fold
        trained_net = None

    # save to output
    all_ytest_df = pd.concat(ytest_df_list, axis=0)
    all_loss_df = pd.concat(loss_df_list, axis=0)
    all_ytest_df.to_csv(args.output_path + '.FNN.cv_' + str(args.cv_int) + '.Prediction.txt', header=True, index=True, sep="\t")
    all_loss_df.to_csv(args.output_path + '.FNN.cv_' + str(args.cv_int) + '.Loss.txt', header=True, index=False, sep="\t")
    if args.shap_bool == True:
        all_shap_df = pd.concat(shap_df_list, axis=0)
        all_shap_df.to_csv(args.output_path + '.FNN.cv_' + str(args.cv_int) + '.SHAP.txt', header=True, index=True, sep="\t")

    # make train/valid loss plots
    tch.save(best_model.state_dict(), args.output_path + '.FNN.cv_' + str(args.cv_int) + 'best_model.pt')
    print( '[Finished in {:}]'.format(myutil.cal_time(datetime.now(), start_time)) )
    # display evaluation metrics of all folds
    mse, rmse, r_square, pccy = mymts.eval_regressor_performance(all_ytest_df, 'resp', 'prediction')
