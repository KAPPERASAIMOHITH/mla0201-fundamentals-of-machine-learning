import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# 1. Data Loading
data = pd.read_csv("HousePricePrediction.csv")

# 2. Data Preprocessing

# One-hot encoding of categorical columns
categorical_cols = ["MSSubClass", "MSZoning", "LotConfig", "BldgType", "Exterior1st"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Drop the ID column
data = data.drop(columns=["Id"])
data.dropna(inplace=True)


# Splitting data
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
# After prediction
df_output = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': predictions})

print(df_output)


