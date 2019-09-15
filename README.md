# Credit-Card-Fraud-Detection-System
This is a machine/ deep learning based project which checks whether a transaction is fraudulent based on the results of previous transactions.

It compares the results of various models like ANN, Random Foreest, Isolation Forest, Local Outlier Factor and XGBoost

The data set is available at https://www.kaggle.com/mlg-ulb/creditcardfraud

# ABSTRACT

The Credit Card Fraud Detection Problem includes modelling past credit card transactions with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not. Our aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.

The dataset we will use for this project will be taken from Kaggle. The dataset contains 300,000 data out of which around 300 data is found to be fraudulent. So we will predict if a new transaction is fraudulent or not. For processing the data, we will shuffle or randomize the data, and use one hot encoding. We will further normalize the data for better understanding and split the data into train and test data.

In order to provide more emphasis on the fraud data in this huge dataset, we will apply a weighting factor on the data. We will create a Deep Learning Model and other machine learning models to predict whether a new transaction is legitimate or not. Finally we will find out the accuracy of the algorithm.
