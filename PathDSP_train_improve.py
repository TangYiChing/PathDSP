import candle
import os
import sys
import datetime
# IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics
#from model_utils.torch_utils import predicting
#import json
#from json import JSONEncoder
from PathDSP_preprocess_improve import cal_time, preprocess, model_preproc_params, app_preproc_params, preprocess_params

#sys.path.append("/usr/local/PathDSP/PathDSP")
#sys.path.append("/usr/local/PathDSP/PathDSP")
#sys.path.append(os.getcwd() + "/PathDSP")
#import FNN_new
import os
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
import myDataloader as mydl
import myUtility as myutil
import polars as pl

file_path = os.path.dirname(os.path.realpath(__file__))

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

# Currently, there are no app-specific args for the train script.
app_train_params = []

# [PathDSP] Model-specific params (Model: PathDSP)
model_train_params = [
    {"name": "cuda_name",  # TODO. frm. How should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    {"name": "learning_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate for the optimizer."
    },
    
]

class RMSELoss(tch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = tch.nn.MSELoss()
        loss = tch.sqrt(criterion(x, y) + eps)
        return loss



def predicting(model, device, data_loader):
    """ Method to make predictions/inference.
    This is used in *train.py and *infer.py

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    data_loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.to(device)
    model.eval()
    total_preds = tch.Tensor()
    total_labels = tch.Tensor()
    print("Make prediction for {} samples...".format(len(data_loader.dataset)))
    with tch.no_grad():
        for i, (data_x, data_y) in enumerate(data_loader):
            data_x, data_y = data_x.to(device), data_y.to(device)
            data_y_pred  = model(data_x)
            # Is this computationally efficient?
            total_preds = tch.cat((total_preds, data_y_pred.cpu()), 0)  # preds to tensor
            total_labels = tch.cat((total_labels, data_y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def predict(net, device, test_dl):
    """
    Return prediction list

    :param net: model
    :param train_dl: train dataloader
    :param device: string representing cpu or cuda:0
    """
    # create result lists
    prediction_list = list()
    true_list = list()

    with tch.no_grad():
        net = net.to(device) # load the network onto the device
        net.eval()
        for i, (X_test, y_test) in enumerate(test_dl):
            X_test, y_test = X_test.to(device), y_test.to(device) # load data onto the device
            y_test_pred  = net(X_test) # test result
            # bring data back to cpu in np.array format, and append to result lists
            prediction_list.append( y_test_pred.cpu().numpy() )
            true_list.append(y_test.cpu().numpy())
            #print(prediction_list)

    # merge all batches
    prediction_list  = np.vstack(prediction_list)
    prediction_list = np.hstack(prediction_list).tolist()
    true_list  = np.vstack(true_list)
    true_list = np.hstack(true_list).tolist()
    # return
    return true_list, prediction_list

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


def fit(net, train_dl, valid_dl, epochs, learning_rate, device, opt_fn, params):
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
    early_stopping = myutil.EarlyStopping(patience=params['patience'], verbose=True, path= params["model_outdir"] + "/checkpoint.pt") # initialize the early_stopping
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
    net.load_state_dict(tch.load(params["model_outdir"] + '/checkpoint.pt'))

    return net, trainloss_list, validloss_list, validr2_list


def run(params):
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")
    params =  preprocess(params)
    
    # set parameters
    myutil.set_seed(params["seed_int"])
    device = myutil.get_device(uth=int(params['cuda_name'].split(':')[1]))
    learning_rate = params['learning_rate']
    epoch = params['epochs']
    batch_size = params['batch_size']
    val_batch = params['val_batch']
    opt_fn = tch.optim.Adam

    # ------------------------------------------------------
    # [PathDSP] Prepare dataloaders
    # ------------------------------------------------------
    print('loadinig data')
    train_df = pl.read_csv(params["train_ml_data_dir"] + "/" + train_data_fname, separator = "\t").to_pandas()
    val_df = pl.read_csv(params["val_ml_data_dir"] + "/" + val_data_fname, separator = "\t").to_pandas()
    Xtrain_arr = train_df.iloc[:, 0:-1].values
    Xvalid_arr = val_df.iloc[:, 0:-1].values
    ytrain_arr = train_df.iloc[:, -1].values
    yvalid_arr = val_df.iloc[:, -1].values    
    Xtrain_arr = np.array(Xtrain_arr).astype('float32')
    Xvalid_arr = np.array(Xvalid_arr).astype('float32')
    ytrain_arr = np.array(ytrain_arr).astype('float32')
    yvalid_arr = np.array(yvalid_arr).astype('float32')
    # create mini-batch
    train_dataset = mydl.NumpyDataset(tch.from_numpy(Xtrain_arr), tch.from_numpy(ytrain_arr))
    valid_dataset = mydl.NumpyDataset(tch.from_numpy(Xvalid_arr), tch.from_numpy(yvalid_arr))
    train_dl = tchud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = tchud.DataLoader(valid_dataset, batch_size=val_batch, shuffle=False)
    
    # ------------------------------------------------------
    # [PathDSP] Prepare model
    # ------------------------------------------------------
    # initial weight
    def init_weights(m):
        if type(m) == tch.nn.Linear:
            tch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    # load model
    n_features = Xtrain_arr.shape[1]
    net = mynet.FNN(n_features)
    net.apply(init_weights)
    
    # ------------------------------------------------------
    # [PathDSP] Training
    # ------------------------------------------------------
    print('start training process')
    trained_net, train_loss_list, valid_loss_list, valid_r2_list = fit(net, train_dl, valid_dl, epoch, learning_rate, device, opt_fn, params)

    loss_df = pd.DataFrame({'epoch':[i+1 for i in range(len(train_loss_list))],
                            'train loss':train_loss_list, 
                            'valid loss': valid_loss_list,
                            'valid r2': valid_r2_list})
    loss_df.to_csv(params['model_outdir'] + '/Val_Loss_orig.txt', header=True, index=False, sep="\t")

    # make train/valid loss plots
    best_model = trained_net
    tch.save(best_model.state_dict(), modelpath)
    #best_model.eval()
    # Compute predictions
    #val_true, val_pred = predicting(best_model, device, valid_dl) # (groud truth), (predictions)
    val_true, val_pred = predict(best_model, device, valid_dl) # (groud truth), (predictions)

    # -----------------------------
    # [Req] Save raw predictions in dataframe
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    frm.store_predictions_df(
        params, y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # -----------------------------
    # [Req] Compute performance scores
    # -----------------------------
    # import ipdb; ipdb.set_trace()
    val_scores = frm.compute_performace_scores(
        params, y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )
    return val_scores


def main(args):
    additional_definitions = model_preproc_params + \
                            model_train_params + \
                            app_train_params
    params = frm.initialize_parameters(
        file_path,
        default_model="PathDSP_default_model.txt",
        #default_model="PathDSP_cs_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1:])
    print("[Training finished in {:}]".format(cal_time(datetime.now(), start)))
