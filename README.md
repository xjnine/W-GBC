# W-GBC
This paper combines multi-granularity cognitive learning and feature weighting information to construct a robust method for generating weighted granular-balls, thus providing a better multi-granularity representation for high-dimensional data.Based on the multi-granularity representation of high-dimensional data, a weighted clustering method that is more adaptive and has better clustering effect is constructed.The effectiveness and accuracy of our proposed method for high-dimensional data characterization is confirmed after extensive testing and comparison on real datasets. This provides a new and effective way to solve the problem of high-dimensional data clustering.

# Files
These program mainly containing:
data: The folder where the dataset is stored;
W_GBC_V1.py: The main function entry point for W-GBC.
W_GBC_split_by_k: The specific implementation process of constructing weighted granules and performing weighted clustering in W-GBC.
wkmeans_no_random.py: The process of obtaining weights based on an improved version of wkmeans.

# Requirements
## Dataset Format
The data folder provides the data used in the experiment. The data and labels are in csv format, with the data file named "X_1_0.csv" and the labels file named "Y_1_0.csv".
## Usage
The parameters input to the main function in WGBC_Test_V1.py are as follows:
    keys: The name of the dataset.
    K: The number of clusters.
The final output includes the evaluation metrics RI, NMI, and ACC.
