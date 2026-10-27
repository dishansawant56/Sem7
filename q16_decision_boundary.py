# Q16: Plot the decision boundary of Linear and RBF SVM
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

import numpy as np, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, kernel in zip(axes, ("linear", "rbf")):
    model = SVC(kernel=kernel).fit(X_train_s, y_train)
    # make a grid of points covering the plot area, predict each one, colour the regions
    xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_test_s[:, 0], X_test_s[:, 1], c=y_test, cmap="coolwarm", edgecolor="k")
    ax.set_title(kernel + " kernel"); ax.set_xlabel("Age (scaled)"); ax.set_ylabel("Salary (scaled)")
plt.show()
print("Linear = straight line boundary. RBF = curved boundary that follows the data.")
