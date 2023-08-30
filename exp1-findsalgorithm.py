import pandas as pd
data = pd.read_csv("enjoysport.csv")
def find_s_algorithm(data):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * (len(data.columns) - 1)
    print("hypothesis",hypothesis);
    
    # Iterate through each positive example
    for idx, row in data.iterrows():
        if row['enjoysport'] == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'
    return hypothesis
specific_hypothesis = find_s_algorithm(data)
print("Specific Hypothesis:", specific_hypothesis)
