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
python ./PathDSP/PathDSP/FNN.py -i inputs.txt -o ./output_prefix
```
# Data preprocessing
Pathway enrichment scores for categorical data (i.e., mutation, copy number variation, and drug targets) were obtained by running the NetPEA algorithm, which is available at: https://github.com/TangYiChing/NetPEA, while pathway enrichment scores for numeric data (i.e., gene expression) was generated with the single-sample Gene Set Enrichment Analsysis (ssGSEA) available here: https://gseapy.readthedocs.io/en/master/gseapy_example.html#3)-command-line-usage-of-single-sample-gseaby 


# Reference
Li, M., Wang, Y., Zheng, R., Shi, X., li,  yaohang, Wu, F., & Wang, J. (2019). DeepDSC: A Deep Learning Method to Predict Drug Sensitivity of Cancer Cell Lines. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 1–1.

Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., … Grobler, J. (2013). API design for machine learning software: experiences from the scikit-learn project. In European Conference on Machine Learning and Principles and Practices of Knowledge Discovery in Databases.
