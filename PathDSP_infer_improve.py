import candle
import os
import sys
#import json
#from json import JSONEncoder
from PathDSP_preprocess_improve import mkdir, preprocess
from PathDSP_train_improve import predicting
import numpy as np
import pandas as pd
from datetime import datetime
import torch as tch
import torch.utils.data as tchud
import polars as pl
import sklearn.metrics as skmts
#sys.path.append("/usr/local/PathDSP/PathDSP")
#sys.path.append("/usr/local/PathDSP/PathDSP")
#sys.path.append(os.getcwd() + "/PathDSP")
import myModel as mynet
import myDataloader as mydl
import myDatasplit as mysplit
import myUtility as myutil

from improve import framework as frm
# from improve.torch_utils import TestbedDataset
from improve.metrics import compute_metrics

from PathDSP_train_improve import (
    preprocess,
    cal_time,
    metrics_list,
    model_preproc_params,
    model_train_params,
)

file_path = os.path.dirname(os.path.realpath(__file__))

# [Req] App-specific params
app_infer_params = []

# [PathDSP] Model-specific params (Model: PathDSP)
model_infer_params = []

def run(params):
    frm.create_outdir(outdir=params["infer_outdir"])
    params =  preprocess(params)
    test_df = pl.read_csv(params['test_data'], separator = "\t").to_pandas()
    Xtest_arr = test_df.iloc[:, 0:-1].values
    ytest_arr = test_df.iloc[:, -1].values
    Xtest_arr = np.array(Xtest_arr).astype('float32')
    ytest_arr = np.array(ytest_arr).astype('float32')
    trained_net = mynet.FNN(Xtest_arr.shape[1])
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"])
    trained_net.load_state_dict(tch.load(modelpath))
    trained_net.eval()
    myutil.set_seed(params["seed_int"])
    device = myutil.get_device(uth=int(params['cuda_name'].split(':')[1]))
    test_dataset = mydl.NumpyDataset(tch.from_numpy(Xtest_arr), tch.from_numpy(ytest_arr))
    test_dl = tchud.DataLoader(test_dataset, batch_size=params['test_batch'], shuffle=False)
    start = datetime.now()
    test_true, test_pred = predicting(trained_net, device, data_loader=test_dl)
    frm.store_predictions_df(
        params, y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )
    test_scores = frm.compute_performace_scores(
        params, y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )
    print('Inference time :[Finished in {:}]'.format(cal_time(datetime.now(), start)))
    return test_scores

def main():
    additional_definitions = model_preproc_params + \
                             model_train_params + \
                             model_infer_params + \
                             app_infer_params
    params = frm.initialize_parameters(
        file_path,
        default_model="PathDSP_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished inference of PathDSP model.")

    
if __name__ == "__main__":
    main()
