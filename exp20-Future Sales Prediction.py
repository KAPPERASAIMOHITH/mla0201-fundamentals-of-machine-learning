import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.io as io
io.renderers.default='browser'

data = pd.read_csv("futuresale prediction.csv")  # Use the correct file path for Android
print(data.head())
print(data.sample(5))
print(data.isnull().sum())

import plotly.express as px
import plotly.graph_objects as go

figure = px.scatter(data_frame=data, x="Sales", y="TV", size="TV", trendline="ols")
figure.show()

figure = px.scatter(data_frame=data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols")
figure.show()

figure = px.scatter(data_frame=data, x="Sales", y="Radio", size="Radio", trendline="ols")
figure.show()

correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

x = np.array(data.drop(["Sales"], axis=1))  # Specify axis=1 to drop the "Sales" column
y = np.array(data["Sales"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

features = np.array([[230.1, 37.8, 69.2]])  # Replace with specific feature values pip install statsmodels,pip uninstall plotly,pip install plotly
print(model.predict(features))
