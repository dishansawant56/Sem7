# Q18: SVM vs Logistic Regression
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

from sklearn.linear_model import LogisticRegression

models = {"Logistic Regression": LogisticRegression(), "SVM (RBF)": SVC(kernel="rbf")}
for name, m in models.items():
    pred = m.fit(X_train_s, y_train).predict(X_test_s)
    print(f"{name:20s} Acc={accuracy_score(y_test, pred):.3f} Prec={precision_score(y_test, pred):.3f} "
          f"Rec={recall_score(y_test, pred):.3f} F1={f1_score(y_test, pred):.3f}")
print("Choose the higher F1. If they are about equal, choose Logistic Regression (simpler).")
