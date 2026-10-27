# Q15: Which matters more - Age, Salary, or both?
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Social_Network_Ads.csv")
y = df["Purchased"]

for name, cols in [("Age only", ["Age"]), ("Salary only", ["EstimatedSalary"]), ("Age + Salary", ["Age", "EstimatedSalary"])]:
    X_train, X_test, y_train, y_test = train_test_split(df[cols], y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_train)
    pred = SVC(kernel="rbf").fit(sc.transform(X_train), y_train).predict(sc.transform(X_test))
    print(f"{name:13s} Acc={accuracy_score(y_test, pred):.3f} Prec={precision_score(y_test, pred):.3f} "
          f"Rec={recall_score(y_test, pred):.3f} F1={f1_score(y_test, pred):.3f}")
print("The option with the best F1/accuracy is the most useful.")
