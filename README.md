# SCD-SAM
* The pytorch implementation for SCD-SAM in paper "SCD-SAM: Adapting Segment Anything Model for Semantic Change Detection in Remote Sensing Imagery".

# Requirements
* Python 3.6
* Pytorch 1.7.0

# Datasets Preparation
The path list in the datasest folder is as follows:

|—train

* ||—A

* ||—B

* ||—labelA

* ||—labelB

|—test

* ||—A

* ||—B

* ||—labelA

* ||—labelB


where A contains pre-temporal images, B contains post-temporal images, labelA contains pre-temporal ground truth images, and labelB contains post-temporal ground truth images.
# Train
* python train.py --dataset-dir dataset-path
# Test
* python eval.py --ckp-paths weight-path --dataset-dir dataset-path
# Visualization
* python visualization visualization.py --ckp-paths weight-path --dataset-dir dataset-path (Note that batch-size must be 1 when using visualization.py)
