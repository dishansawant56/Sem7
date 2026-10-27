# Q1: Which combination of TV / Radio / Newspaper predicts Sales best?
import pandas as pd, itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")
print(df.head()); print(df.isnull().sum())          # inspect + missing values

y = df["Sales"]
cols = ["TV", "Radio", "Newspaper"]
X_train, X_test, y_train, y_test = train_test_split(df[cols], y, test_size=0.2, random_state=42)

for k in (1, 2, 3):                                  # try 1, 2, then 3 columns
    for combo in itertools.combinations(cols, k):
        combo = list(combo)
        model = LinearRegression().fit(X_train[combo], y_train)
        pred = model.predict(X_test[combo])
        rmse = mean_squared_error(y_test, pred) ** 0.5
        print(combo, "RMSE =", round(rmse, 3), " R2 =", round(r2_score(y_test, pred), 3))

print("Answer: the combination with the LOWEST RMSE and HIGHEST R2 is best.")
