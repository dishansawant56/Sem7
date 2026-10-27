# Q17: False positives and false negatives
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

pred = SVC(kernel="rbf").fit(X_train_s, y_train).predict(X_test_s)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("True Positives :", tp, "(predicted buy, did buy)")
print("True Negatives :", tn, "(predicted no, did not buy)")
print("False Positives:", fp, "(predicted buy, but did NOT buy) -> wasted advertising money")
print("False Negatives:", fn, "(predicted no, but DID buy)      -> lost customers")
print("Precision:", round(precision_score(y_test, pred), 3), " Recall:", round(recall_score(y_test, pred), 3))
