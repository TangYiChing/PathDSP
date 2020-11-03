"""
Given pretrained model
To make predictions
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
    early_stopping = myutil.EarlyStopping(patience=10, verbose=True) # initialize the early_stopping
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
        print('epoch={:}/{:}, train loss={:.5f}, valid loss={:.5f}'.format(
               epoch+1, epochs, train_epoch_loss / len(train_dl),
                                valid_epoch_loss / len(valid_dl)))

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    net.load_state_dict(tch.load('checkpoint.pt'))

    return  net, trainloss_list, validloss_list

def train(net, train_dl,  epochs, learning_rate, device, opt_fn):
    """
    Return train loss and trained model
    """
    ## setup
    criterion = RMSELoss() #tch.nn.MSELoss() # setup LOSS function
    optimizer = opt_fn(net.parameters(), lr=learning_rate, weight_decay=1e-5) # setup optimizer
    net = net.to(device) # load the network onto the device
    trainloss_list = [] # metrics: MSE, size equals to EPOCH
    # repeat the training for EPOCH times
    for epoch in range(epochs):
        ## training phase
        net.train()
        # initial loss
        train_epoch_loss = 0.0 # save loss for each epoch, batch by batch
        for i, (X_train, y_train) in enumerate(train_dl):
            X_train, y_train = X_train.to(device), y_train.to(device) # load data onto the device
            y_train_pred  = net(X_train) # train result
            #train_loss = criterion(y_train_pred, y_train.unsqueeze(1).float()) # calculate loss
            train_loss = criterion(y_train_pred, y_train.float())
            optimizer.zero_grad() # clear gradients
            train_loss.backward() # backpropagation
            #### add this if you have gradient explosion problem ###
            clip_value = 5
            tch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            ###   climp gradient within -5 ~ 5  ####################
            optimizer.step() # update weights
            train_epoch_loss += train_loss.item() # adding loss from each batch
        # calculate total loss of all batches
        trainloss_list.append( train_epoch_loss / len(train_dl) )
        # display print message
        #print('epoch={:}/{:}, train loss={:.5f}'.format(
        #       epoch+1, epochs, train_epoch_loss / len(train_dl)))
    return net

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
    return prediction_list

# define arguments
def parse_parameter():
    parser = argparse.ArgumentParser(description = "Load pretrained model and make prediction")
    parser.add_argument("-train", "--train_path",
                        required = True,
                        help = "train path")
    parser.add_argument("-test", "--test_path",
                        required = True,
                        help = "test path")
    parser.add_argument("-p", "--pretrained_path",
                        required = True,
                        help = "path to pretrained model")
    parser.add_argument("-s", "--seed_int",
                        required = False,
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-g", "--gpu_int",
                        default = 0,
                        type = int,
                        help = "assign the n-th GPU")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "output path")
    return parser.parse_args()

if __name__ == "__main__":
    start_time = datetime.now()

    # get args
    args = parse_parameter()

    # set parameters
    myutil.set_seed(args.seed_int)
    device = myutil.get_device(uth=args.gpu_int)
    learning_rate = 0.0004
    epoch = 800
    batch_size = 8
    opt_fn = tch.optim.Adam

    # load data
    train_df = pd.read_csv(args.train_path, header=0, index_col=[0,1], sep="\t")
    test_df = pd.read_csv(args.test_path, header=0, index_col=[0,1], sep="\t")
    trainX_df = train_df.iloc[:, 0:-1]
    trainy_df = train_df.iloc[:, -1]
    testX_df = test_df.iloc[:, 0:-1]
    testy_df = test_df.iloc[:, -1]

    # convert to numpy array
    Xtrain_arr = trainX_df.values.astype('float32')
    ytrain_arr = trainy_df.values.astype('float32')
    Xtest_arr = testX_df.values.astype('float32')
    ytest_arr = testy_df.values.astype('float32')
    Xtrain_arr, Xvalid_arr, ytrain_arr, yvalid_arr = skms.train_test_split(Xtrain_arr, ytrain_arr,
                                                                           test_size=0.1, random_state=args.seed_int)

    # create mini-batch
    train_dataset = mydl.NumpyDataset(tch.from_numpy(Xtrain_arr), tch.from_numpy(ytrain_arr))
    valid_dataset = mydl.NumpyDataset(tch.from_numpy(Xvalid_arr), tch.from_numpy(yvalid_arr))
    test_dataset = mydl.NumpyDataset(tch.from_numpy(Xtest_arr), tch.from_numpy(ytest_arr))
    train_dl = tchud.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dl = tchud.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dl = tchud.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # output
    ytest_df = testy_df.to_frame()
    
    # load model
    n_features = Xtest_arr.shape[1]
    trained_net = mynet.FNN(n_features)
    trained_net.load_state_dict(tch.load(args.pretrained_path))
    ##initial weight
    #def init_weights(m):
    #    if type(m) == tch.nn.Linear:
    #        tch.nn.init.kaiming_uniform_(m.weight)
    #        m.bias.data.fill_(0.01)
    #trained_net.apply(init_weights)

    # train on whole data
    #final_net = train(trained_net, train_dl,  epoch, learning_rate, device, opt_fn)
    final_net, train_loss_list, valid_loss_list = fit(trained_net, train_dl, valid_dl, epoch, learning_rate, device, opt_fn)
    # test the  model
    prediction_list = predict(final_net, test_dl, device)
    
    # append to result 
    ytest_df['prediction'] = prediction_list
    ytest_df.to_csv(args.output_path + '.FNN.Pretrained.Prediction.txt', header=True, index=True, sep="\t")
  
    # evaluation metrics
    mse = skmts.mean_squared_error(ytest_arr, prediction_list)
    rmse = np.sqrt(mse)
    r_square = skmts.r2_score(ytest_arr, prediction_list)
    pcc, pval = scistat.pearsonr(ytest_arr, prediction_list)
    print('MSE={:}, RMSE={:}, R2={:}, PCC={:}'.format(
           mse, rmse, r_square, pcc))

    print( '[Finished in {:}]'.format(myutil.cal_time(datetime.now(), start_time)) )
