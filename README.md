# Requirements 

1. A CUDA-enabled GPU is required to build the RF pulse simulation C++ code. 
2. The libtorch C++ library must be downloaded. 

# Installation 

1. Install necessary python libraries via: ```pip3 install -r requirements.txt```
2. Open ```setup.py``` and modify line 6 to be the path to the downloaded libtorch C++ library. 
3. Build the RF pulse simulation extension via: ```python3 setup.py build_ext --inplace```

# Reproducing Results 

Separate Jupyter notebooks are provided to optimize the pulses, evaluate their performance, and create the figures. Respectively, these are named as: 

1. ```optimize_pulses.ipynb```
2. ```evaluate_pulses.ipynb```
3. ```plot_pulses.ipynb```
