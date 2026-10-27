# Q5: Effect of training data size (50%, 70%, 80%) on performance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")
X = df[["TV", "Radio", "Newspaper"]]; y = df["Sales"]

# keep the SAME test set for all sizes so the comparison is fair
pool_X, X_test, pool_y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for frac in (0.5, 0.7, 0.8):
    k = int(frac * len(df))                      # number of training rows
    model = LinearRegression().fit(pool_X.iloc[:k], pool_y.iloc[:k])
    pred = model.predict(X_test)
    print(f"Train size {int(frac*100)}% ({k} rows): RMSE =", round(mean_squared_error(y_test, pred) ** 0.5, 3),
          " R2 =", round(r2_score(y_test, pred), 3))

print("More data usually helps a little, then the improvement flattens (simple model).")
