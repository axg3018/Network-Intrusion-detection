import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from time import time


start_time = time()
#Import dataset
dataset = pd.read_csv('attacks_normal.csv')
data = dataset.iloc[1:, 1:] 
X = data.iloc[:, :-1].values 
y = data.iloc[:, 41].values 

#normal = true = 1, attack = false = 0;
y[y==11] = 1
y[y!=1] = 0
#split data intp train and test, fit, transform
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#classify using k-nn
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Print results
print("Accuracy Score:", accuracy_score(y_test, y_pred)) 
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred, target_names=['attack', 'normal']))
end_time = time()
print("\n\nTime taken: %.2f" %(end_time-start_time),"seconds")
