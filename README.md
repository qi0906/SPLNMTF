# SPLNMTF

This repository has been created to present the recent work done on drug repositioning.
Our paper: Improved Computational Drug-Repositioning by Self-Paced Non-Negative Matrix Tri-Factorization.

## What can you find in this repository ?

This repository contains data, scripts and results related to our recent work.

In particular, you will find:

- 3 folders, 
[***data***](data/) which stores all kinds of raw data;
[***cross_val_data***](cross_val_data/) which stores the 5-fold cross-validation results;
[***results***](results/) which stores the prediction results.


- 1 *.py* files, 
[***Metric.py***](Metric.py) which contains a variety of model evaluation metrics(AUC,AUPR,MSE,RMSE).


- 11 other *.m* files, 
[***main.m***](main.m): which is the main function, the program starts running;
[***SPLNMTF.m***](SPLNMTF.m): Function for the algorithm of SPLLNMTF;
[***matrix_initialization.m***](matrix_initialization.m): Function for initializing G matrices by using different strategies;
[***block_matrices.m***](block_matrices.m): Function for computing block matrices from edgelists;
[***centric_rule.m***](centric_rule.m): Function for exporting significant values from the reconstructed matrix by using row-centric or colomn-centric rule or combination (mix) of these two;
[***compute_clusters_ssnmtf.m***](compute_clusters_ssnmtf.m): Function for assigning entities to clusters;
[***concat_data.m***](concat_data.m): Function for compliting matrices;
[***connectivity.m***](connectivity.m): Function for creating connectivity matrix from matrix factor H;
[***cross_validation.m***](cross_validatio.m): Function for 5-fold cross-validation;
[***export_significant_associations.m***](export_significant_associations.m): Function for reconstructing relation matrices;
[***run_SPLNMTF.m***](run_SPLNMTF.m): Function for exporting predictions.



## How to run ?

If you want to run these files, you may need to download the version of matlab is matlab R2016a, the version of python is Python 3.10.2 and install the following packages:
*sklearn, matplotlib, numpy, pandas, csv etc.


