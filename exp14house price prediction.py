import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv("HousePricePrediction.csv")
categorical_cols = ["MSSubClass", "MSZoning", "LotConfig", "BldgType", "Exterior1st"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data = data.drop(columns=["Id"])
data.dropna(inplace=True)
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
df_output = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': predictions})
print(df_output)


