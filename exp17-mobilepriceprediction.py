import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("mobile_prices.csv")
X = data.drop(columns=['price_range'])  # Features
y = data['price_range']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
new_mobile_features = [[1500, 1, 2.2, 1, 5, 1, 32, 0.8, 140, 8, 6, 800, 1200, 4096, 12, 7, 15, 1, 1, 1]]
predicted_price_range = model.predict(new_mobile_features)
print(f"Predicted Price Range for the New Mobile: {predicted_price_range[0]:.2f}")
