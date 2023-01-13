# PathDSP
Explainable Drug Sensitivity Prediction through Cancer Pathway Enrichment Scores

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

Where input.txt should be in the input format shown above. Example input file can be found at https://zenodo.org/record/7532963
```
# Data preprocessing
Pathway enrichment scores for categorical data (i.e., mutation, copy number variation, and drug targets) were obtained by running the NetPEA algorithm, which is available at: https://github.com/TangYiChing/NetPEA, while pathway enrichment scores for numeric data (i.e., gene expression) was generated with the single-sample Gene Set Enrichment Analsysis (ssGSEA) available here: https://gseapy.readthedocs.io/en/master/gseapy_example.html#3)-command-line-usage-of-single-sample-gseaby 


# Reference
Tang, Y.-C., & Gottlieb, A. (2021). Explainable drug sensitivity prediction through cancer pathway enrichment. Scientific Reports, 11(1), 3128. https://doi.org/10.1038/s41598-021-82612-7
