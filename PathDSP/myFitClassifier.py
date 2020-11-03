"""
"""


import numpy as np
import torch as tch
import pandas as pd
import sklearn.metrics as skmts
from pandas.core.common import flatten


def train_mul(model, train_dl,  epochs, learning_rate, device, opt_fn):
    """
    Return train loss and trained model

    NOTE:
    if binary, criterion = tch.nn.BCELoss()
    if multi-class, criteiron = tch.nn.CrossEntropyLoss()
    """
    ## setup
    criterion = tch.nn.CrossEntropyLoss() #loss_fn # setup LOSS function
    optimizer = opt_fn(model.parameters(), lr=learning_rate, weight_decay=1e-5) # setup optimizer
    model = model.to(device) # load the network onto the device
    model.train()
    # repeat the training for EPOCH times
    loss_list = [] # metrics: MSE, size equals to EPOCH
    for epoch in range(epochs):
        # initial loss
        epoch_loss = 0.0 # save loss for each epoch, batch by batch
        epoch_auc = 0.0  # save acc for each epoch, batch by batch
        # train model with batches
        for i, (X_train, y_train) in enumerate(train_dl):
            X_train, y_train = X_train.to(device), y_train.to(device) # load data onto the device
            optimizer.zero_grad() # clear gradients
            y_pred  = model(X_train) # train result
            loss = criterion(y_pred, y_train.squeeze_()) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # update weights
            epoch_loss += loss.item() # adding loss from each batch
        # calculate total loss of all batches
        loss_list.append( epoch_loss / len(train_dl) )
        # display print message
        print('epoch={:}/{:} | train loss={:.5f} '.format(
               epoch+1, epochs, epoch_loss / len(train_dl)))
 
def predict_mul(model, test_dl, device):
    """
    Return prediction list
    """
    model.eval()
    with tch.no_grad():
        y_pred_list, y_pred_proba_df_list = [], []
        for i, (X_test, y_test) in enumerate(test_dl):
            X_test, y_test = X_test.to(device), y_test.to(device) # load data onto the device
            y_pred_proba  = model(X_test) # test result
            y_pred = tch.log_softmax(y_pred_proba, dim=1)
            _, y_pred = tch.max(y_pred, dim=1)
            # append to result
            y_pred_proba_df_list.append(pd.DataFrame(y_pred_proba.cpu().numpy()))
            y_pred_list.append(y_pred.cpu().numpy().tolist())
    # return
    prediction_list = list(flatten(y_pred_list))
    prediction_probability_df = pd.concat(y_pred_proba_df_list, axis=0)
    return prediction_list, prediction_probability_df

def train(model, train_dl,  epochs, learning_rate, device, opt_fn):
    """
    Return train loss and trained model

    NOTE: 
    if binary, criterion = tch.nn.BCELoss()
    if multi-class, criteiron = tch.nn.CrossEntropyLoss()
    """
    ## setup
    criterion = tch.nn.BCELoss() #loss_fn # setup LOSS function
    optimizer = opt_fn(model.parameters(), lr=learning_rate, weight_decay=1e-5) # setup optimizer
    model = model.to(device) # load the network onto the device
    model.train()
    # repeat the training for EPOCH times
    loss_list = [] # metrics: MSE, size equals to EPOCH
    for epoch in range(epochs):
        # initial loss
        epoch_loss = 0.0 # save loss for each epoch, batch by batch
        epoch_auc = 0.0  # save acc for each epoch, batch by batch
        # train model with batches
        for i, (X_train, y_train) in enumerate(train_dl):
            X_train, y_train = X_train.to(device), y_train.to(device) # load data onto the device
            optimizer.zero_grad() # clear gradients
            y_pred  = model(X_train) # train result
            loss = criterion(y_pred, y_train.float()) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # update weights
            epoch_loss += loss.item() # adding loss from each batch
        # calculate total loss of all batches
        loss_list.append( epoch_loss / len(train_dl) )
        # display print message
        print('epoch={:}/{:} | train loss={:.5f} '.format(
               epoch+1, epochs, epoch_loss / len(train_dl)))

def predict(model, test_dl, device):
    """
    Return prediction list
    """
    model.eval()
    with tch.no_grad():
        y_pred_list, y_pred_proba_list = [], []
        for i, (X_test, y_test) in enumerate(test_dl):
            X_test, y_test = X_test.to(device), y_test.to(device) # load data onto the device
            y_pred_proba  = model(X_test) # test result
            y_pred = tch.round(y_pred_proba)
            # append to result
            y_pred_proba_list.append(y_pred_proba.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())
            
    # return
    prediction_list, prediction_probability_list = np.vstack(y_pred_list), np.vstack(y_pred_proba_list)
    return prediction_list, prediction_probability_list
