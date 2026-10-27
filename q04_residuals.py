# Q4: Residual analysis - find the 5 biggest errors and plot Actual vs Predicted
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Advertising.csv")
X_train, X_test, y_train, y_test = train_test_split(df[["TV", "Radio", "Newspaper"]], df["Sales"], test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

result = X_test.copy()
result["Actual"] = y_test
result["Predicted"] = pred
result["Residual"] = result["Actual"] - result["Predicted"]     # residual = actual - predicted
result["AbsResidual"] = result["Residual"].abs()
print(result.sort_values("AbsResidual", ascending=False).head(5))

plt.scatter(y_test, pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")   # perfect line
plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales"); plt.title("Actual vs Predicted")
plt.show()
print("Why big errors? Non-linear effects, outliers, extreme spends, noise, missing features.")
