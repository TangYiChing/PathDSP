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


# Reference
Li, M., Wang, Y., Zheng, R., Shi, X., li,  yaohang, Wu, F., & Wang, J. (2019). DeepDSC: A Deep Learning Method to Predict Drug Sensitivity of Cancer Cell Lines. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 1–1.
