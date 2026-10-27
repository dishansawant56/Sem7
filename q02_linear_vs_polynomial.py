# Q2: Linear vs Polynomial (degree 2) regression using TV and Radio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")
X = df[["TV", "Radio"]]; y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normal linear model
lin = LinearRegression().fit(X_train, y_train)
p1 = lin.predict(X_test)

# polynomial: make new columns TV^2, Radio^2, TV*Radio first
poly = PolynomialFeatures(degree=2, include_bias=False)
Xp_train = poly.fit_transform(X_train)
Xp_test = poly.transform(X_test)          # transform only (do not fit) on test
pm = LinearRegression().fit(Xp_train, y_train)
p2 = pm.predict(Xp_test)

print("Linear     RMSE:", round(mean_squared_error(y_test, p1) ** 0.5, 3), " R2:", round(r2_score(y_test, p1), 3))
print("Polynomial RMSE:", round(mean_squared_error(y_test, p2) ** 0.5, 3), " R2:", round(r2_score(y_test, p2), 3))
print("If polynomial is better -> TV and Radio work together (interaction / non-linear effect).")
