# RADE Sci-Kit Classifier (v1.0)

RADE is a resource-efficient decision tree ensemble method (DTEM) based anomaly 
detection approach that augments standard DTEM classifiers resulting in 
competitive anomaly detection capabilities and significant savings in resource 
usage.

The current implementation of RADE augments either Random-Forest or XGBoost.

More information about RADE can be found in:<br/>
arXiv: [RADE: Resource-Efficient Supervised Anomaly Detection Using Decision Tree-Based Ensemble Methods](https://arxiv.org/pdf/1909.11877.pdf){target="_blank"}

## Files:

#### rade_classifier.py 
RADE sci-kit classifier

#### example_program.py
Basic comparison example between RF, XGBoost, and RADE

#### requirments.txt
Install CMake 3.13 or higher.<br/>
Then, to install RADE's prerequisities, run:<br/>
$ pip3 install -r requirments.txt

## Prerequisities:
numpy<br/>
pandas<br/>
sklearn<br/>
xgboost (CMake 3.13 or higher is required)<br/>


For more information, support and advanced examples contact:<br/>
Yaniv Ben-Itzhak, [ybenitzhak@vmware.com](mailto:ybenitzhak@vmware.com)<br/>
Shay Vargaftik, [shayv@vmware.com](mailto:shayv@vmware.com)<br/>
