# Q11: Random Forest feature importance, then retrain with the top 3 features
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.lower()
df = df.rename(columns={"kms_driven": "km_driven"})
df = df[["year", "selling_price", "km_driven", "transmission", "owner"]].dropna()
df["owner"] = df["owner"].astype(str)

y = df["selling_price"]
X = pd.get_dummies(df.drop(columns="selling_price"), drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)
importance.plot(kind="barh"); plt.title("Feature importance"); plt.show()

top3 = list(importance.index[:3])
rf3 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train[top3], y_train)
for name, model, cols in [("All features", rf, list(X.columns)), ("Top 3 only", rf3, top3)]:
    pred = model.predict(X_test[cols])
    print(name, "RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3), " R2:", round(r2_score(y_test, pred), 3))
