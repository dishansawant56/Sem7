# Q10: Find outliers in selling_price with IQR, remove them from TRAIN, compare
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.lower()
df = df.rename(columns={"kms_driven": "km_driven"})
df = df[["year", "selling_price", "km_driven", "transmission", "owner"]].dropna()
df["owner"] = df["owner"].astype(str)
df = pd.get_dummies(df, drop_first=True)

train, test = train_test_split(df, test_size=0.2, random_state=42)

# IQR rule
q1 = train["selling_price"].quantile(0.25)
q3 = train["selling_price"].quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("Allowed price range:", round(low, 2), "to", round(high, 2))
clean = train[(train["selling_price"] >= low) & (train["selling_price"] <= high)]
print("Outliers removed:", len(train) - len(clean))

plt.boxplot(train["selling_price"]); plt.title("Selling price boxplot"); plt.show()

X_test = test.drop(columns="selling_price"); y_test = test["selling_price"]   # test stays untouched
for name, data in [("With outliers   ", train), ("Without outliers", clean)]:
    m = LinearRegression().fit(data.drop(columns="selling_price"), data["selling_price"])
    pred = m.predict(X_test)
    print(name, "RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3), " R2:", round(r2_score(y_test, pred), 3))
