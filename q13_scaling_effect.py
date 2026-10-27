# Q13: Effect of scaling on RBF SVM
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

no_scale = SVC(kernel="rbf").fit(X_train, y_train).predict(X_test)          # raw data
scaled = SVC(kernel="rbf").fit(X_train_s, y_train).predict(X_test_s)        # standardised data
for name, pred in [("Without scaling", no_scale), ("With scaling   ", scaled)]:
    print(name, "Accuracy:", round(accuracy_score(y_test, pred), 3), " F1:", round(f1_score(y_test, pred), 3))
print("Why? Salary (~50,000) is much bigger than Age (~40), so without scaling salary dominates the distance and Age is ignored.")
