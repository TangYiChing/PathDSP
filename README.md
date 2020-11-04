# PathDSP
Explainable Drug Sensitivity Prediction through Cancer Pathway Enrichment Scores

# Requirments

# Input format

|drug|cell|feature1|....|feature2|resp|
|----|----|--------|----|--------|----|
|5-FU|03|0|....|0.02|-2.3|
|5-FU|23|1|....|0.04|-3.4|

# Usage:
```python
# run FNN 
python FNN.py -i inputs.txt -o ./output_prefix
```
# Data preprocessing
Pathway enrichment scores for categorical data (i.e., mutation, copy number variation, and drug targets) were obtained by the NetPEA algorithm, which is available at: https://github.com/TangYiChing/NetPEA, while pathway enrichment scores for numeric data (i.e., gene expression) was obtained by the gseapy tool (https://gseapy.readthedocs.io/en/master/gseapy_example.html).

step1. performe 


# Reference
Li, M., Wang, Y., Zheng, R., Shi, X., li,  yaohang, Wu, F., & Wang, J. (2019). DeepDSC: A Deep Learning Method to Predict Drug Sensitivity of Cancer Cell Lines. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 1–1.
