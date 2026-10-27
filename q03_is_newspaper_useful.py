# Q3: Does Newspaper add useful information?
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")
X_train, X_test, y_train, y_test = train_test_split(df[["TV", "Radio", "Newspaper"]], df["Sales"], test_size=0.2, random_state=42)
n = len(y_test)

def adjusted_r2(r2, n, k):                 # k = number of features
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

for name, cols in [("With Newspaper   ", ["TV", "Radio", "Newspaper"]), ("Without Newspaper", ["TV", "Radio"])]:
    pred = LinearRegression().fit(X_train[cols], y_train).predict(X_test[cols])
    r2 = r2_score(y_test, pred)
    print(name, "RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3),
          " R2:", round(r2, 4), " Adj R2:", round(adjusted_r2(r2, n, len(cols)), 4))

print("If the numbers barely change (or Adj R2 falls) -> Newspaper is NOT useful.")
