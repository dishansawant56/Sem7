# Q9: Effect of encoding categorical columns (transmission, owner)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.lower()
df = df.rename(columns={"kms_driven": "km_driven"})
df = df[["year", "selling_price", "km_driven", "transmission", "owner"]].dropna()
df["owner"] = df["owner"].astype(str)

y = df["selling_price"]
encoded = pd.get_dummies(df.drop(columns="selling_price"), drop_first=True)   # one-hot encoding
print("Columns after encoding:", list(encoded.columns))
X_train, X_test, y_train, y_test = train_test_split(encoded, y, test_size=0.2, random_state=42)

numeric = ["year", "km_driven"]
for name, cols in [("Numeric only          ", numeric), ("Numeric + encoded cols", list(encoded.columns))]:
    pred = LinearRegression().fit(X_train[cols], y_train).predict(X_test[cols])
    print(name, "RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3), " R2:", round(r2_score(y_test, pred), 3))
print("Encoding lets the model use text columns; drop_first=True avoids the dummy variable trap.")
