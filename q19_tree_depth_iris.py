# Q19: Effect of tree depth on overfitting (Iris)
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

depths = [2, 3, 4, 5]
train_acc, test_acc = [], []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(tree.score(X_train, y_train))
    test_acc.append(tree.score(X_test, y_test))
    print("depth", d, "-> train accuracy:", round(train_acc[-1], 3), " test accuracy:", round(test_acc[-1], 3))

plt.plot(depths, train_acc, "o-", label="Train"); plt.plot(depths, test_acc, "s-", label="Test")
plt.xlabel("max_depth"); plt.ylabel("Accuracy"); plt.legend(); plt.show()
print("Overfitting = train accuracy high but test accuracy lower. Pick the depth with best test accuracy.")
