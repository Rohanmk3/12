import numpy as np
import pandas as pd
from sklearn import metrics

df=pd.read_csv(r"C:\Users\HPR\Desktop\ML Syllabus\Play Tennis.csv")
value=['Outlook','Temprature','Humidity','Wind']
df

len(df)

df.shape

df.head()

df.tail() 

df.describe() 

from sklearn import preprocessing
string_to_int= preprocessing.LabelEncoder() #encode your data
df=df.apply(string_to_int.fit_transform) #fit and transform it
df 


feature_cols = ['Outlook','Temprature','Humidity','Wind']
X = df[feature_cols ] #contains the attribute
y = df.Play_Tennis

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 

from sklearn.tree import DecisionTreeClassifier # import the classifier
classifier =DecisionTreeClassifier(criterion="entropy", random_state=100) # create a classifier object
classifier.fit(X_train, y_train) 


y_pred= classifier.predict(X_test) 

from sklearn.metrics import accuracy_score
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

data_p=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
data_p

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
