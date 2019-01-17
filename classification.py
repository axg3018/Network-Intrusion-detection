
# classification using data set 
#Importing the required library 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from time import time
  


start_time = time()
#Importing the dataset 
dataset = pd.read_csv('attacks_normal.csv')
data = dataset.iloc[1:, 1:] 
norm = 11
attacks = ['Back', 'BufferOverflow', 'FTPWrite', 'GuessPassword', 'Imap', 'IPSweep', 'Land', 'Normal']
#seperating the predicting column from the whole dataset 
X = data.iloc[:, :-1].values 
y = data.iloc[:, 41].values 
  
#Encoding the predicting variable 
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y) 


#Spliting the data into test and train dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
 

#Using the random forest classifier for the prediction 
classifier=RandomForestClassifier() 
classifier=classifier.fit(X_train,y_train) 
predicted=classifier.predict(X_test) 
  
#printing the results 

print ('Confusion Matrix :')
cm = confusion_matrix(y_test, predicted)
print(cm) 
print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
print ('Report : ') 
print (classification_report(y_test, predicted, target_names=attacks)) 
end_time = time()
print("\n\nTime taken: %.2f" %(end_time-start_time),"seconds")






