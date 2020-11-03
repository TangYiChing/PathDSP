"""
Return torch dataset Given file path to a cell by PPI gene probability dataframe
"""


import numpy as np
import pandas as pd
import torch as tch
import torch.utils.data as tchud
import sklearn.model_selection as skms
import sklearn.preprocessing as skpre
import myDatasplit as mysplit

class CategoricalDataset(tchud.Dataset):
    """
    Return torch dataset
    """
    # load data from file
    def __init__(self, fin):
        # load data
        self.df = pd.read_csv(fin, header=0, index_col=[0,1], sep="\t")
        # separate features and labels
        self.X = self.df.iloc[:, :-1].astype('float32').values # feautes
        self.y = self.df.iloc[:, -1].astype('float32').values # labels
        #self.y = self.y.reshape((len(self.y), 1)) # ensure target has the right shape
        # label encode target and ensure the values are floats
        self.y = skpre.LabelEncoder().fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index], self.y[index]]

class NumpyDataset(tchud.Dataset):
    """
    Return torch dataset, Given X and y numpy array
    """
    def __init__(self, X_arr, y_arr):
        self.X = X_arr
        self.y = y_arr
        # label encode target and ensure the values are floats
        #self.y = skpre.LabelEncoder().fit_transform(self.y)
        self.y = self.y.reshape((len(self.y),1))
     
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)

class DataframeDataset(tchud.Dataset):
    """
    Return torch dataset Given file path to a cell by PPI gene encoded dataframe.

    """
    # load data from file
    def __init__(self, fin):
        # load data
        self.df = pd.read_csv(fin, header=0, index_col=[0,1], sep="\t")
        # separate features and labels
        self.X = self.df.iloc[:, :-1].astype('float32').values # feautes
        self.y = self.df.iloc[:, -1].astype('float32').values # labels
        self.y = self.y.reshape((len(self.y), 1)) # ensure target has the right shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index], self.y[index], index]

    def get_index_names(self, index):
        return self.df.iloc[index].index.tolist()

    def get_splits(self, n_test=0.33):
        """return disjoint tch.Dataset"""
        test_size = round(n_test*len(self.X))
        train_size = len(self.X) - test_size
        return tchud.random_split(self, [train_size, test_size])

    def get_splits_cv(self, cv=10):
        """return list of kfsplits of tch.Dataset with size equals to cv"""
        k = cv
        kf = skms.KFold(n_splits=int(k)) # use scikit's KFold
        # save splits to a list
        return_list = []
        for n_fold, (train_index, test_index) in enumerate(kf.split(self.X)):
            # get disjoint splits for each fold
            Xtrain_array = self.X[train_index]
            ytrain_array = self.y[train_index]
            Xtest_array = self.X[test_index]
            ytest_array = self.y[test_index]
            # tchud.TensorDataset didn't work for numpy array, so split into 2 steps
            # 1. to tensor
            Xtrain_tensor = tch.from_numpy(Xtrain_array)
            ytrain_tensor = tch.from_numpy(ytrain_array)
            Xtest_tensor = tch.from_numpy(Xtest_array)
            ytest_tensor = tch.from_numpy(ytest_array)
            # 2. to Dataset
            train_dataset = tchud.TensorDataset(Xtrain_tensor, ytrain_tensor)
            test_dataset = tchud.TensorDataset(Xtest_tensor, ytest_tensor)
            # append to result
            return_list.append( (train_dataset, test_dataset) )
        return return_list

if __name__ == "__main__":
    print('initiating an dataset....')
    fin = './example_input/CCLE.fnn.input.txt'
    dataset = DataframeDataset(fin)
    print('option 1: one-time train/test split....')
    train, test = dataset.get_splits(n_test=0.2)
    train_dl = tchud.DataLoader(train, batch_size=8, shuffle=True)
    test_dl = tchud.DataLoader(test, batch_size=8, shuffle=False)
    print('calling function get_splits(n_test=0.2)....')
    print('    n_test={:}, train={:}, test={:}'.format(0.2, len(train_dl), len(test_dl)))
    
    print('option 2: cross validation splits....')
    print('calling function get_splits_cv(cv=3)....')
    for n_fold, (train, test) in enumerate(dataset.get_splits_cv(cv=3)):
        train_dl = tchud.DataLoader(train, batch_size=8, shuffle=True)
        test_dl = tchud.DataLoader(test, batch_size=8, shuffle=False)
        print('    cv={:}, fold={:}, train={:}, test={:}'.format(
                   3, n_fold, len(train_dl), len(test_dl)))

    print('option 3: one-time train/valid/test split....')
    split = mysplit.DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=8, num_workers=1)
    print('train={:}, valid={:}, test={:}'.format(
           len(train_loader), len(val_loader), len(test_loader)))
