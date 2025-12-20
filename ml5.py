import numpy as np
import pandas as pd
from sklearn import metrics

df = pd.read_csv(r"C:\Users\HPR\Desktop\ML Syllabus\Play_Tennis_reg.csv")

len(df)
df.shape

X = df.drop("Golf Players", axis=1)
y = df['Golf Players']

X
y

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

string_to_int = preprocessing.LabelEncoder()
X = X.apply(string_to_int.fit_transform)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg = reg.fit(X, y)

y_pred = reg.predict([[2,1,0,1]])
print("Result is:", y_pred)

y_pred = reg.predict([[2,1,0,0]])
print("Result is:", y_pred)

y_pred = reg.predict([[1,2,0,0]])
print("Result is:", y_pred)
