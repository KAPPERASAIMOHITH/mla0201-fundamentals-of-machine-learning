import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("IRIS.csv")
X = data.drop(columns=['species'])  # Features
y = data['species']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Perceptron(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your data
predicted_species = model.predict(new_data)
print(f"Predicted Species: {predicted_species[0]}")
