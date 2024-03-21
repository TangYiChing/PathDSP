# PathDSP
Explainable Drug Sensitivity Prediction through Cancer Pathway Enrichment Scores

# Download benchmark data

Download the cross-study analysis (CSA) benchmark data into the model directory from https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/

```
mkdir process_dir
cd process_dir
wget --cut-dirs=7 -P ./ -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
```

Benchmark data will be downloaded under `process_dir/csa_data/`

# Example usage with Conda

Download PathDSP and IMPROVE

```
mkdir repo
cd repo
git clone -b develop https://github.com/JDACS4C-IMPROVE/PathDSP.git
git clone -b develop https://github.com/JDACS4C-IMPROVE/IMPROVE.git
```

# Download author data

```
cd ../
mkdir author_data
bash repo/PathDSP/download_author_data.sh author_data/
```

Author data will be downloaded under `process_dir/author_data/`
PathDSP will be installed at `process_dir/repo/PathDSP`
IMPROVE will be installed at `process_dir/repo/IMPROVE`

Create environment

```
cd repo/PathDSP/
conda env create -f environment_082223.yml -n PathDSP_env
```

Activate environment

```
conda activate PathDSP_env
```

Install CANDLE package

```
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

Define enviroment variabels

```
improve_lib="/path/to/IMPROVE/repo/"
pathdsp_lib="/path/to/pathdsp/repo/"
# notice the extra PathDSP folder after pathdsp_lib
export PYTHONPATH=$PYTHONPATH:${improve_lib}:${pathdsp_lib}/PathDSP/
export IMPROVE_DATA_DIR="/path/to/csa_data/"
export AUTHOR_DATA_DIR="/path/to/author_data/"
```

Perform preprocessing step

```
# go two upper level
cd ../../
python repo/PathDSP/PathDSP_preprocess_improve.py
```

Train the model

```
python repo/PathDSP/PathDSP_train_improve.py
```

Metrics regarding validation scores is located at: `${train_ml_data_dir}/val_scores.json`
Final trained model is located at: `${train_ml_data_dir}/model.pt`. Parameter definitions can be found at `process_dir/repo/PathDSP/PathDSP_default_model.txt`

Perform inference on the testing data

```
python repo/PathDSP/PathDSP_infer_improve.py
```

Metrics regarding test process is located at: `${infer_outdir}/test_scores.json`
Final prediction on testing data is located at: `${infer_outdir}/test_y_data_predicted.csv`

# Example usage with singularity container

# Download benchmark data

Download the cross-study analysis (CSA) benchmark data into the model directory from https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/

```
mkdir process_dir
cd process_dir
wget --cut-dirs=7 -P ./ -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
```

# Download author data

Download model specific data under csa_data/ directory

```
git clone -b develop https://github.com/JDACS4C-IMPROVE/PathDSP.git
bash PathDSP/download_author_data.sh csa_data/
```

Setup Singularity

```
git clone -b develop https://github.com/JDACS4C-IMPROVE/Singularity.git
cd Singularity
./setup
source config/improve.env
```

Build Singularity from definition file

```
singularity build --fakeroot PathDSP.sif definitions/PathDSP.def
```

Perform preprocessing using csa benchmarking data

```
singularity exec --nv --bind ${IMPROVE_DATA_DIR}:/candle_data_dir PathDSP.sif preprocess.sh /candle_data_dir --ml_data_outdir /candle_data_dir/preprocess_data/
```

Train the model

```
singularity exec --nv --bind ${IMPROVE_DATA_DIR}:/candle_data_dir PathDSP.sif train.sh 0 /candle_data_dir --train_ml_data_dir /candle_data_dir/preprocess_data/ --val_ml_data_dir /candle_data_dir/preprocess_data/ --model_outdir /candle_data_dir/out_model/
```

Metrics regarding validation scores is located at: `${train_ml_data_dir}/val_scores.json`
Final trained model is located at: `${train_ml_data_dir}/model.pt`. 

Perform inference on the testing data

```
singularity exec --nv --bind ${IMPROVE_DATA_DIR}:/candle_data_dir PathDSP.sif infer.sh 0 /candle_data_dir --test_ml_data_dir /candle_data_dir/preprocess_data/ --model_dir /candle_data_dir/out_model/ --infer_outdir /candle_data_dir/out_infer/
```

Metrics regarding test process is located at: `${infer_outdir}/test_scores.json`
Final prediction on testing data is located at: `${infer_outdir}/test_y_data_predicted.csv`


# Docs from original authors (below)

# Requirments

# Input format

|drug|cell|feature_1|....|feature_n|drug_response|
|----|----|--------|----|--------|----|
|5-FU|03|0|....|0.02|-2.3|
|5-FU|23|1|....|0.04|-3.4|

Where feature_1 to feature_n are the pathway enrichment scores and the chemical fingerprint coming from data processing
# Usage:
```python
# run FNN 
python ./PathDSP/PathDSP/FNN.py -i input.txt -o ./output_prefix

Where input.txt should be in the input format shown above. 
Example input file can be found at https://zenodo.org/record/7532963
```
# Data preprocessing
Pathway enrichment scores for categorical data (i.e., mutation, copy number variation, and drug targets) were obtained by running the NetPEA algorithm, which is available at: https://github.com/TangYiChing/NetPEA, while pathway enrichment scores for numeric data (i.e., gene expression) was generated with the single-sample Gene Set Enrichment Analsysis (ssGSEA) available here: https://gseapy.readthedocs.io/en/master/gseapy_example.html#3)-command-line-usage-of-single-sample-gseaby 


# Reference
Tang, Y.-C., & Gottlieb, A. (2021). Explainable drug sensitivity prediction through cancer pathway enrichment. Scientific Reports, 11(1), 3128. https://doi.org/10.1038/s41598-021-82612-7