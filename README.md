# Baseball-Pitch-Prediction
This repository contains code used to define and train a Recurrent Neural Network Model used to predict type of baseball pitch in at bat
using the Pytorch Deep Learning framework.  

Data is found on kaggle at https://www.kaggle.com/pschale/mlb-pitch-data-20152018.  
EDA and descriptive figures found in MLB_Pitching_EDA.ipynb file  
The files used to create formatted hdf5 datasets are:  
* clean_data.py for hidden state at bat initialization architecture dataset
* clean_data_seq.py for at bat info sequence vector architecture
* clean_data_SCHERZER.py for Max Scherzer at bats ONLY (6 pitch types)  

The model folder contains the network classes (net.py and net_seq.py) and the data loading classes (data_loader.py and data_loader_seq.py)  

Train and save them model using various changes to the dataset and network architecture using the files below:
* train.py for training on full dataset using architecture where initial hidden state is initialized using at bat information
* train_seq.py for training on full dataset with initial at bat info being provided in every vector of pitch sequence
* train_scherzer.py for training and predicting pitches in at bats thrown by pitcher Max Scherzer ONLY  

Results figures and scores are created and found in resultsAnalysis.ipynb file in results folder.
