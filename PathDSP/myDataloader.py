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

class NumpyDataset(tchud.Dataset):
    """
    Return torch dataset, Given X and y numpy array
    """
    def __init__(self, X_arr, y_arr):
        self.X = X_arr
        self.y = y_arr
        self.y = self.y.reshape((len(self.y),1))
     
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
