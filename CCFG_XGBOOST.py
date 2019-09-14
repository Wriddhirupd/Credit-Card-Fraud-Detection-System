# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:37:22 2019

@author: Wriddhirup Dutta
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:40:42 2019

@author: Wriddhirup Dutta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('D:/6TH_SEMESTER/CSE3019 DATA MINING/PROJECT/Fraud_Detection_Model/creditcard.csv')
#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, 4].values

shuffled_data = dataset.sample(frac=1)
# Change Class column into Class_0 ([1 0] for legit data) and Class_1 ([0 1] for fraudulent data)
#one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])
# Change all values into numbers between 0 and 1
normalized_data = (shuffled_data - shuffled_data.min()) / (shuffled_data.max() - shuffled_data.min())
# Store just columns V1 through V28 in df_X and columns Class_0 and Class_1 in df_y
df_X = normalized_data.drop(['Class'], axis=1)
df_y = normalized_data['Class']
# Convert both data_frames into np arrays of float32
ar_X, ar_y = np.asarray(df_X.values, dtype='float32'), np.asarray(df_y.values, dtype='float32')

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(ar_X, ar_y, test_size = 0.2, random_state = 0)

# Feature Scaling

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set

classifier = XGBClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
cm
acc = accuracy_score(y_test,y_pred)
acc



