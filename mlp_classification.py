
# classification using data set 
#Importing the required library 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
directory = os.getcwd()
directory = os.path.join(directory, "optimized_attacks_normal")
os.chdir(directory)
dataset = pd.read_csv('attacks_normal.csv')
data = dataset.iloc[1:, 1:] 

norm = 11
attacks = ['Back', 'BufferOverflow', 'FTPWrite', 'GuessPassword', 'Imap', 'Normal/Other attacks']
#seperating the predicting column from the whole dataset 
X = data.iloc[:, :-1].values 
y = data.iloc[:, 41].values 
y[y > 4] = 11  
#Encoding the predicting variable 
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y) 

#Spliting the data into test and train dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33) 
 
#Using the mlp classifier for the prediction 
classifier=MLPClassifier(solver='lbfgs', random_state=0)
classifier=classifier.fit(X_train,y_train) 
predicted=classifier.predict(X_test) 
  
#printing the results 

print ('Confusion Matrix :')
cm = confusion_matrix(y_test, predicted)
print(cm) 
print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
print ('Report : ') 
print (classification_report(y_test, predicted, target_names=attacks)) 






