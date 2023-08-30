import candle
import os
import sys
#import json
#from json import JSONEncoder
from preprocess_new import mkdir, preprocess
import numpy as np
import pandas as pd
from datetime import datetime
import torch as tch
import torch.utils.data as tchud
import polars as pl
import sklearn.metrics as skmts
#sys.path.append("/usr/local/PathDSP/PathDSP")
sys.path.append("/usr/local/PathDSP/PathDSP")
import FNN_new


file_path = os.path.dirname(os.path.realpath(__file__))
required = None
additional_definitions = None


# initialize class
class PathDSP_candle(candle.Benchmark):
    def set_locals(self):
        '''
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        '''
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

def initialize_parameters():
    preprocessor_bmk = PathDSP_candle(file_path,
        'PathDSP_params.txt',
        'pytorch',
        prog='PathDSP_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

    
def run(params):
    trained_net = FNN_new.mynet.FNN(Xtest_arr.shape[1])
    trained_net.load_state_dict(tch.load(params['data_dir'] + '/model.pt'))
    trained_net.eval()
    test_df = pl.read_csv(params['test_data'], separator = "\t").to_pandas()
    FNN_new.myutil.set_seed(params["seed_int"])
    device = FNN_new.myutil.get_device(uth=params["gpu_int"])
    Xtest_arr = test_df.iloc[:, 0:-1].values
    ytest_arr = test_df.iloc[:, -1].values
    Xtest_arr = np.array(Xtest_arr).astype('float32')
    ytest_arr = np.array(ytest_arr).astype('float32')
    test_dataset = FNN_new.mydl.NumpyDataset(tch.from_numpy(Xtest_arr), tch.from_numpy(ytest_arr))
    test_dl = tchud.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    start = datetime.now()
    prediction_list = FNN_new.predict(trained_net, test_dl, device)
    print('Inference time :[Finished in {:}]'.format(FNN_new.cal_time(datetime.now(), start)))
    # evaluation metrics
    mse = skmts.mean_squared_error(ytest_arr, prediction_list)
    rmse = np.sqrt(mse)
    r2_pred = FNN_new.r2_score(ytest_arr, prediction_list)
    loss_pred = pd.DataFrame({'metric': ['rmse', 'r2'],
                              'value': [rmse, r2_pred]})
    loss_pred.to_csv(params['data_dir'] + '/Loss_pred.txt', header=True, index=False, sep="\t")
    ytest_df = test_df.iloc[:, -1].to_frame()
    ytest_df['prediction'] = prediction_list
    ytest_df.to_csv(params['data_dir'] + '/Prediction.txt', header=True, index=True, sep="\t")


def candle_main():
    params = initialize_parameters()
    data_dir = os.environ['CANDLE_DATA_DIR'] + '/' + '/Data/'
    params =  preprocess(params, data_dir)
    run(params)
    
if __name__ == "__main__":
    candle_main()
