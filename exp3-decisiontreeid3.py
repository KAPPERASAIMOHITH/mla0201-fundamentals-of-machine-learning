import numpy as np
import pandas as pd
from scipy.stats import mode

# Define the dataset
data = pd.read_csv("enjoysport.csv")
df = pd.DataFrame(data)

def ID3(data):
    # Base case: if all examples have the same label, return that label
    if len(data['enjoysport'].unique()) == 1:
        return data['enjoysport'].iloc[0]
    
    # Base case: if no features left, return the majority label
    if len(data.columns) == 1:
        return mode(data['enjoysport'])[0][0]
    
    # Find the best feature to split on (simplified: first available feature)
    best_feature = data.columns[0]
    
    # Create a new node
    tree = {best_feature: {}}
    
    # Recursively build the tree
    unique_values = data[best_feature].unique()
    for value in unique_values:
        subtree = ID3(data[data[best_feature] == value].drop(columns=[best_feature]))
        tree[best_feature][value] = subtree
    
    return tree

def classify(tree, sample):
    feature = list(tree.keys())[0]
    value = sample[feature]
    
    if value in tree[feature]:
        if isinstance(tree[feature][value], dict):
            return classify(tree[feature][value], sample)
        else:
            return tree[feature][value]
    else:
        return mode(df['enjoysport'])[0][0]

# Build the tree and classify a new sample
decision_tree = ID3(df)
new_sample = {
    'sky': 'sunny',
    'airtemp': 'warm',
    'humidity': 'high',
    'wind': 'strong',
    'water': 'warm',
    'forcast': 'same'
}
classification_result = classify(decision_tree, new_sample)
print("Classification Result:", classification_result)
