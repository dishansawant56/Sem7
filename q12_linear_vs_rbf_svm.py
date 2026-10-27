# Q12: Linear SVM vs RBF SVM
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

for kernel in ("linear", "rbf"):
    pred = SVC(kernel=kernel).fit(X_train_s, y_train).predict(X_test_s)
    print("\nKernel:", kernel)
    print(confusion_matrix(y_test, pred))
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall   :", round(recall_score(y_test, pred), 3))
    print("F1       :", round(f1_score(y_test, pred), 3))
print("\nRBF is better if the data is not separable by a straight line.")
