# Q7: Create 'vehicle age' from year and compare km_driven alone vs km_driven + age
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.lower()
df = df.rename(columns={"kms_driven": "km_driven"}).dropna()

df["age"] = 2020 - df["year"]            # new feature (use the reference year the question gives)
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(df[["km_driven", "age"]], y, test_size=0.2, random_state=42)

for name, cols in [("km_driven only", ["km_driven"]), ("km_driven + age", ["km_driven", "age"])]:
    model = LinearRegression().fit(X_train[cols], y_train)
    pred = model.predict(X_test[cols])
    print(name, "-> RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3), " R2:", round(r2_score(y_test, pred), 3))

print("Age usually improves the model a lot: older car = lower price.")
