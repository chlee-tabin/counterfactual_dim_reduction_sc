# A causal single-cell analysis framework
This work is currently underway.
An incomplete description of the work can be found [here](https://github.com/hanbin973/counterfactual_dim_reduction_sc/raw/main/Counterfactual_Dimension_Reduction_and_Feature_Selection.pdf).
Some claims in the draft are yet verified.
We are planning to write a [scanpy](https://scanpy.readthedocs.io/en/stable/) implementation. 

Our method (and also the implementation) is extremely fast and efficient.
It fully supports `scipy` `sparse` functionality and never invokes a large dense matrix.
For example, PCA is performed by calculating the covariance matrix of the residual matrix without actually calculating the residual matrix.

Please [e-mail](hanbin973@snu.ac.kr) me if you use our method for your work.
Use it at your own risk.

