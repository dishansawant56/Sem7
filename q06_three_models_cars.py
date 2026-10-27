# Q6: Linear Regression vs Decision Tree vs Random Forest (used cars)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.lower()
df = df.rename(columns={"kms_driven": "km_driven"})
df = df[["year", "selling_price", "km_driven", "transmission", "owner"]].drop_duplicates().dropna()
df["owner"] = df["owner"].astype(str)
print(df.head()); print(df.isnull().sum())

y = df["selling_price"]
X = pd.get_dummies(df.drop(columns="selling_price"), drop_first=True)   # text -> numbers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {"Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(random_state=42),
          "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)}
for name, m in models.items():
    pred = m.fit(X_train, y_train).predict(X_test)
    print(f"{name:18s} RMSE = {mean_squared_error(y_test, pred) ** 0.5:.3f}   R2 = {r2_score(y_test, pred):.3f}")

print("Best model = lowest RMSE / highest R2 (usually Random Forest).")
