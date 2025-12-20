import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Iris.csv")
print(data.shape)

data = data.drop('Id', axis=1)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Shape of X is %s and shape of y is %s" % (X.shape, y.shape))

total_classes = y.nunique()
print("Number of unique species in dataset are:", total_classes)

print(y.value_counts())

X_train, X_val, Y_train, Y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

adb = AdaBoostClassifier()
adb_model = adb.fit(X_train, Y_train)

print("The accuracy of the model on validation set is", adb_model.score(X_val, Y_val))

y_pred = adb_model.predict(X_val)
accuracy = accuracy_score(Y_val, y_pred)
print(f"The Accuracy of Prediction on Iris Flower is: {accuracy}")

df = pd.DataFrame({'Actual': Y_val, 'Predicted': y_pred})
print(df)
