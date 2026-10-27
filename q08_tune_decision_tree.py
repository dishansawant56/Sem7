# Q8: Tune a Decision Tree (max_depth, min_samples_split, min_samples_leaf) with GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.lower()
df = df.rename(columns={"kms_driven": "km_driven"})
df = df[["year", "selling_price", "km_driven", "transmission", "owner"]].dropna()
df["owner"] = df["owner"].astype(str)

y = df["selling_price"]
X = pd.get_dummies(df.drop(columns="selling_price"), drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# before tuning
base = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
print("Before tuning -> RMSE:", round(mean_squared_error(y_test, base.predict(X_test)) ** 0.5, 3),
      " R2:", round(r2_score(y_test, base.predict(X_test)), 3))

# search the best settings
grid = {"max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]}
search = GridSearchCV(DecisionTreeRegressor(random_state=42), grid, cv=5, scoring="neg_root_mean_squared_error")
search.fit(X_train, y_train)
print("Best settings:", search.best_params_)

best = search.best_estimator_
print("After tuning  -> RMSE:", round(mean_squared_error(y_test, best.predict(X_test)) ** 0.5, 3),
      " R2:", round(r2_score(y_test, best.predict(X_test)), 3))
