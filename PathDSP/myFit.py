"""
"""


import numpy as np
import torch as tch
import pandas as pd

class RMSELoss(tch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = tch.nn.MSELoss()
        loss = tch.sqrt(criterion(x, y) + eps)
        return loss

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
        print('epoch={:}/{:}, train loss={:.5f}'.format(
               epoch+1, epochs, train_epoch_loss / len(train_dl)))
    #print(net.hidden1.weight)
    # convert to loss dataframe
    train_loss_df = pd.DataFrame({'Epoch': [i+1 for i in range(len(trainloss_list))],
                                  'Train Loss': trainloss_list})
    # return
    return train_loss_df, net

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
    criterion = tch.nn.MSELoss() # setup LOSS function
    optimizer = opt_fn(net.parameters(), lr=learning_rate, weight_decay=1e-5) # setup optimizer
    net = net.to(device) # load the network onto the device
    trainloss_list = [] # metrics: MSE, size equals to EPOCH
    validloss_list = [] # metrics: MSE, size equals to EPOCH
    # repeat the training for EPOCH times
    for epoch in range(epochs):
        ## training phase
        net.train()
        # initial loss
        train_epoch_loss = 0.0 # save loss for each epoch, batch by batch
        for i, (X_train, y_train) in enumerate(train_dl):
            X_train, y_train = X_train.to(device), y_train.to(device) # load data onto the device
            y_train_pred  = net(X_train) # train result
            train_loss = criterion(y_train_pred, y_train.unsqueeze(1)) # calculate loss
            print('type={:}/{:}, loss={:}'.format(type(y_train_pred), type(y_train.unsqueeze(1)), train_loss))
            optimizer.zero_grad() # clear gradients
            train_loss.backward() # backpropagation
            #### add this if you have gradient explosion problem ###
            clip_value = 5
            tch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            ########climp gradient within -5 ~ 5 ###################
            optimizer.step() # update weights
            train_epoch_loss += train_loss.item() # adding loss from each batch
        # calculate total loss of all batches
        trainloss_list.append( train_epoch_loss / len(train_dl) )

        ## validation phase
        with tch.no_grad():
            net.eval()
            valid_epoch_loss = 0.0 # save loss for each epoch, batch by batch
            for i, (X_valid, y_valid) in enumerate(valid_dl):
                X_valid, y_valid = X_valid.to(device), y_valid.to(device) # load data onto the device
                y_valid_pred  = net(X_valid) # valid result
                valid_loss = criterion(y_valid_pred, y_valid.unsqueeze(1)) # calculate loss
                valid_epoch_loss += valid_loss.item() # adding loss from each batch
        # calculate total loss of all batches, and append to result list
        validloss_list.append( valid_epoch_loss / len(valid_dl) )

        # display print message
        print('epoch={:}/{:}, train loss={:.5f}, valid loss={:.5f}'.format(
               epoch+1, epochs, train_epoch_loss / len(train_dl),
                                valid_epoch_loss / len(valid_dl)))
    # convert to loss dataframe
    trainvalid_loss_df = pd.DataFrame({'Epoch': [i+1 for i in range(len(trainloss_list))], 
                                       'Train Loss': trainloss_list, 
                                       'Valid Loss': validloss_list})
    trainvalid_loss_df = trainvalid_loss_df.set_index('Epoch')
    # return
    return trainvalid_loss_df, net

def predict(net, test_dl, device):
    """
    Return prediction list

    :param net: model
    :param train_dl: train dataloader
    :param device: string representing cpu or cuda:0
    """
    # load model
    #model.load_state_dict(tch.load(filepath))
    #net = net.to(device) # load the network onto the device
    # create result lists
    prediction_list = list()
  
    with tch.no_grad():
        net.eval() 
        for i, (X_test, y_test) in enumerate(test_dl):
            X_test, y_test = X_test.to(device), y_test.to(device) # load data onto the device
            y_test_pred  = net(X_test) # test result
            # bring data back to cpu in np.array format, and append to result lists
            prediction_list.append( y_test_pred.cpu().numpy() )
            #index_list.append( index_test.cpu().numpy() )
            #print(prediction_list)
     
    # merge all batches 
    #prediction_list = [item.squeeze().tolist() for item in prediction_list]
    #prediction_list = [item for li in prediction_list for item in li]
    #index_list = [item.squeeze().tolist() for item in index_list]
    #index_list = [item for li in index_list for item in li]
    prediction_list  = np.vstack(prediction_list)
    prediction_list = np.hstack(prediction_list).tolist()
   
    # return
    return prediction_list
