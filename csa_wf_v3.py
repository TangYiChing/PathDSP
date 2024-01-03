""" Python implementation of cross-study analysis workflow """
# cuda_name = "cuda:6"
cuda_name = "cuda:7"

import os
import subprocess
import warnings
from time import time
from pathlib import Path

import pandas as pd

# IMPROVE imports
from improve import framework as frm
# import improve_utils
# from improve_utils import improve_globals as ig

# GraphDRP imports
# TODO: change this for your model
import PathDSP_preprocess_improve
import PathDSP_train_improve
import PathDSP_preprocess_improve

# from ap_utils.classlogger import Logger
# from ap_utils.utils import get_print_func, Timer


class Timer:
  """ Measure time. """
  def __init__(self):
    self.start = time()

  def timer_end(self):
    self.end = time()
    return self.end - self.start

  def display_timer(self, print_fn=print):
    time_diff = self.timer_end()
    if time_diff // 3600 > 0:
        print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
    else:
        print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )


fdir = Path(__file__).resolve().parent

y_col_name = "auc"
# y_col_name = "auc1"

maindir = Path(f"./{y_col_name}")
MAIN_ML_DATA_DIR = Path(f"./{maindir}/ml.data")
MAIN_MODEL_DIR = Path(f"./{maindir}/models")
MAIN_INFER_OUTDIR = Path(f"./{maindir}/infer")

# Check that environment variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception("ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n")
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

params = frm.initialize_parameters(
    fdir,
    default_model="csa_workflow_params.txt",
)

main_datadir = Path(os.environ["IMPROVE_DATA_DIR"])
raw_datadir = main_datadir / params["raw_data_dir"]
x_datadir = raw_datadir / params["x_data_dir"]
y_datadir = raw_datadir / params["y_data_dir"]
splits_dir = raw_datadir / params["splits_dir"]

# lg = Logger(main_datadir/"csa.log")
print_fn = print
# print_fn = get_print_func(lg.logger)
print_fn(f"File path: {fdir}")

### Source and target data sources
## Set 1 - full analysis
# source_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
## Set 2 - smaller datasets
# source_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# source_datasets = ["GDSCv1", "CTRPv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
## Set 3 - full analysis for a single source
# source_datasets = ["CCLE"]
# source_datasets = ["CTRPv2"]
source_datasets = ["GDSCv1"]
target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "gCSI", "GDSCv2"]
## Set 4 - same source and target
# source_datasets = ["CCLE"]
# target_datasets = ["CCLE"]
## Set 5 - single source and target
# source_datasets = ["GDSCv1"]
# target_datasets = ["CCLE"]

# only_cross_study = False
only_cross_study = True


## Splits
# split_nums = []  # all splits
# split_nums = [0]
# split_nums = [4, 7]
split_nums = [1, 4, 7]
# split_nums = [1, 3, 5, 7, 9]

## Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
# epochs = 2
# epochs = 30
# epochs = 50
epochs = 70
# epochs = 100
# epochs = 150
# config_file_name = "csa_params.txt"
# config_file_path = fdir/config_file_name

def build_split_fname(source, split, phasea):
    """ Build split file name. If file does not exist continue """
    return f"{source_data_name}_split_{split}_{phase}.txt"

# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================

timer = Timer()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
print_fn(f"\nsource_datasets: {source_datasets}")
print_fn(f"target_datasets: {target_datasets}")
print_fn(f"split_nums:      {split_nums}")
# import pdb; pdb.set_trace()
for source_data_name in source_datasets:

    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((splits_dir).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
        # num_splits = 1
    else:
        # Use the specified splits
        split_files = []
        for s in split_nums:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]

    # --------------------
    # Preprocess and Train
    # --------------------
    # import pdb; pdb.set_trace()
    for split in split_nums:
        print_fn(f"Split id {split} out of {len(split_nums)} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        # split = 11
        # files_joined = [str(s) for s in split_files]
        # TODO: check this!
        for phase in ["train", "val", "test"]:
            fname = build_split_fname(source_data_name, split, phase)
            # print(f"{phase}: {fname}")
            if fname not in "\t".join(files_joined):
                warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                continue

        # import pdb; pdb.set_trace()
        for target_data_name in target_datasets:
            if only_cross_study and (source_data_name == target_data_name):
                continue # only cross-study
            print_fn(f"\nSource data: {source_data_name}")
            print_fn(f"Target data: {target_data_name}")

            # EXP_ML_DATA_DIR = ig.ml_data_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            ml_data_outdir = MAIN_ML_DATA_DIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            # import pdb; pdb.set_trace()
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
            timer_preprocess = Timer()
            # ml_data_path = graphdrp_preprocess_improve.main([
            #     "--train_split_file", f"{source_data_name}_split_{split}_train.txt",
            #     "--val_split_file", f"{source_data_name}_split_{split}_val.txt",
            #     "--test_split_file", str(test_split_file_name),
            #     "--ml_data_outdir", str(ml_data_outdir),
            #     "--y_col_name", y_col_name
            # ])
            print_fn("\nPreprocessing")
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            # test_split_file = f"{source_data_name}_split_{split}_test.txt"
            print_fn(f"train_split_file: {train_split_file}")
            print_fn(f"val_split_file:   {val_split_file}")
            print_fn(f"test_split_file:  {test_split_file}")
            print_fn(f"ml_data_outdir:   {ml_data_outdir}")
            # import pdb; pdb.set_trace()
            preprocess_run = ["python",
                  "PathDSP_preprocess_improve.py",
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  "--ml_data_outdir", str(ml_data_outdir),
                  "--y_col_name", str(y_col_name)
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)
            # print(result.stdout)
            # print(result.stderr)
            timer_preprocess.display_timer(print_fn)

            # p2 (p1): Train model
            # Train a single model for a given [source, split] pair
            # Train using train samples and early stop using val samples
            # import pdb; pdb.set_trace()
            model_outdir = MAIN_MODEL_DIR/f"{source_data_name}"/f"split_{split}"
            if model_outdir.exists() is False:
                train_ml_data_dir = ml_data_outdir
                val_ml_data_dir = ml_data_outdir
                timer_train = Timer()
                # graphdrp_train_improve.main([
                #     "--train_ml_data_dir", str(train_ml_data_dir),
                #     "--val_ml_data_dir", str(val_ml_data_dir),
                #     "--model_outdir", str(model_outdir),
                #     "--epochs", str(epochs),  # available in config_file
                #     # "--ckpt_directory", str(MODEL_OUTDIR),  # TODO: we'll use candle known param ckpt_directory instead of model_outdir
                #     # "--cuda_name", "cuda:5"
                # ])
                print_fn("\nTrain")
                print_fn(f"train_ml_data_dir: {train_ml_data_dir}")
                print_fn(f"val_ml_data_dir:   {val_ml_data_dir}")
                print_fn(f"model_outdir:      {model_outdir}")
                # import pdb; pdb.set_trace()
                train_run = ["python",
                      "PathDSP_train_improve.py",
                      "--train_ml_data_dir", str(train_ml_data_dir),
                      "--val_ml_data_dir", str(val_ml_data_dir),
                      "--model_outdir", str(model_outdir),
                      "--epochs", str(epochs),
                      "--cuda_name", cuda_name,
                      "--y_col_name", y_col_name
                ]
                result = subprocess.run(train_run, capture_output=True,
                                        text=True, check=True)
                # print(result.stdout)
                # print(result.stderr)
                timer_train.display_timer(print_fn)

            # Infer
            # p3 (p1, p2): Inference
            # import pdb; pdb.set_trace()
            test_ml_data_dir = ml_data_outdir
            model_dir = model_outdir
            infer_outdir = MAIN_INFER_OUTDIR/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            timer_infer = Timer()
            # graphdrp_infer_improve.main([
            #     "--test_ml_data_dir", str(test_ml_data_dir),
            #     "--model_dir", str(model_dir),
            #     "--infer_outdir", str(infer_outdir),
            #     # "--cuda_name", "cuda:5"
            # ])
            print_fn("\nInfer")
            print_fn(f"test_ml_data_dir: {test_ml_data_dir}")
            #print_fn(f"val_ml_data_dir:  {val_ml_data_dir}")
            print_fn(f"infer_outdir:     {infer_outdir}")
            # import pdb; pdb.set_trace()
            infer_run = ["python",
                  "PathDSP_infer_improve.py",
                  "--test_ml_data_dir", str(test_ml_data_dir),
                  "--model_dir", str(model_dir),
                  "--infer_outdir", str(infer_outdir),
                  "--cuda_name", cuda_name,
                  "--y_col_name", y_col_name
            ]
            result = subprocess.run(infer_run, capture_output=True,
                                    text=True, check=True)
            timer_infer.display_timer(print_fn)

timer.display_timer(print_fn)
print_fn("Finished a full cross-study run.")
