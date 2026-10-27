# Q20: Gini vs Entropy on Iris, with tree plots
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, criterion in zip(axes, ("gini", "entropy")):
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=42).fit(X_train, y_train)
    pred = tree.predict(X_test)
    print("\nCriterion:", criterion, " Accuracy:", round(accuracy_score(y_test, pred), 3))
    print(confusion_matrix(y_test, pred))
    plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, ax=ax)
    ax.set_title(criterion)
plt.show()
print("Both measure impurity of a node; results are usually very similar.")
