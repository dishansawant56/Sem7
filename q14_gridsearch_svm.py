# Q14: GridSearchCV to tune C and gamma of RBF SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)       # learn scaling from TRAIN only
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

from sklearn.model_selection import GridSearchCV

default = SVC(kernel="rbf").fit(X_train_s, y_train).predict(X_test_s)
print("Default settings -> Accuracy:", round(accuracy_score(y_test, default), 3), " F1:", round(f1_score(y_test, default), 3))

grid = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}
search = GridSearchCV(SVC(kernel="rbf"), grid, cv=5, scoring="accuracy").fit(X_train_s, y_train)
print("Best C and gamma:", search.best_params_, " (CV accuracy:", round(search.best_score_, 3), ")")

tuned = search.predict(X_test_s)
print("Tuned settings   -> Accuracy:", round(accuracy_score(y_test, tuned), 3), " F1:", round(f1_score(y_test, tuned), 3))
