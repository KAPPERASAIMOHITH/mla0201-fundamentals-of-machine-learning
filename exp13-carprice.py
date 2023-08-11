import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Importing the dataset
data = pd.read_csv("C:/Users/koppo/Downloads/CarPrice.csv")

# Data Exploration
data.head()
data.shape
data.isnull().sum() # Checking if the dataset has NULL Values
data.info()
data.describe()
data.CarName.unique()

# Drop the 'CarName' column
data = data.drop(columns=['CarName'])

# Convert categorical variables into numerical format using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Prepare data for training
predict = "price"
X = data.drop([predict], axis=1)
y = data[predict]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Car Price Prediction Model using DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model using Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Example usage: Predicting the price for a new car
new_car_features = pd.DataFrame({
    'symboling': [4],
    'wheelbase': [97.2],
    'carlength': [172.0],
    'carwidth': [65.4],
    'carheight': [54.3],
    'curbweight': [2330],
    'enginesize': [109],
    'boreratio': [3.19],
    'stroke': [3.03],
    'compressionratio': [9.0],
    'horsepower': [85],
    'peakrpm': [5800],
    'citympg': [27],
    'highwaympg': [32],
    'fueltype_gas': [1],  # Example for fueltype: 'gas', you can set other columns based on your data
    # Add other columns as required based on your data
})

# Align the new_car_features with the training data to ensure the same columns
new_car_features_aligned = new_car_features.reindex(columns=X.columns, fill_value=0)

# Making predictions on the new car data
predicted_price = model.predict(new_car_features_aligned)
print("Predicted Price for the new car:", predicted_price[0])
