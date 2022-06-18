# Point clouds registration algorithms - Find the best correspondences set

The goal of this project is to develop a benchmark for point clouds registration algorithms and
compare between the in several error indices to find out who find the best correspondences set and
to reach conclusions about the problem and the algorithms.


## Pre Project 

First we build the dataset for this project by using - 
https://github.com/iralabdisco/point_clouds_registration_benchmark
(follow the Instructions to get the ETH Dataset )

    
## How to use:
### Dependencies
- Python3
- [open3d](http://www.open3d.org/)
- [OT](https://pythonot.github.io/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [torch](https://pytorch.org/)
- [dgl](https://www.dgl.ai/)
- [matplotlib](https://matplotlib.org/)


### Instructions

#### - The dataset must be in Dataset/eth/ dir (follow point_clouds_registration_benchmark instructions)
#### - all tests and references is under References dir. 
#### - just need to ensure that the data is under Dataset/eth/ dir and run the test file.
#### - the refernces file save images for regestration process in Images/eth/_NAME_OF_SPESIFIC_METHOD_ dir
#### - the sensitivity file save graphes of regestration process in GRAPHES/_NAME_OF_SPESIFIC_METHOD_ dir
#### - the Experiment Tests dir include images from custom test
#### - the Report dir include doc (~6000 words) about our result and conclusions.