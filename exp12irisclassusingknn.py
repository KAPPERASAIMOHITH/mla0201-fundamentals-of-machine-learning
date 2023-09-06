from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
predictions = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
new_flower_features = [[5.1, 3.5, 1.4, 0.2]]  # Replace with the features of the new flower
predicted_species = knn_classifier.predict(new_flower_features)
print("Predicted Species:", iris.target_names[predicted_species])
