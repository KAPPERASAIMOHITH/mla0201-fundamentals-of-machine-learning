import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Load the dataset (replace with your dataset)
data = pd.read_csv("IRIS.csv")

# Assuming you have preprocessed the data and split it into X (features) and y (target)
X = data.drop(columns=['species'])  # Features
y = data['species']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Perceptron model
model = Perceptron(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy to evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example new data for prediction
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your data
predicted_species = model.predict(new_data)
print(f"Predicted Species: {predicted_species[0]}")
