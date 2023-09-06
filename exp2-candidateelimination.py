import pandas as pd
data = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\dat.csv")
def initialize_hypothesis(data):
    specific_hypothesis=['0']*(len(data.columns)-1)
    general_hypothesis=['?']*(len(data.columns)-1)
    return specific_hypothesis,general_hypothesis
def canditate_elimination(data):
    specific_hypothesis,general_hypothesis=initialize_hypothesis(data)
    for idx,row in data.iterrows():
        if row['enjoysport']=='yes':
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i]=='0':
                    specific_hypothesis[i]=row[i]
                elif specific_hypothesis[i]!=row[i]:
                    specific_hypothesis[i]='?'
        else:
            for i in range(len(general_hypothesis)):
                if specific_hypothesis[i]!=row[i]:
                    general_hypothesis[i]='?'
    return specific_hypothesis,general_hypothesis
specific,general=canditate_elimination(data)
print("specific hypothesis:",specific)
print("general hypothesis:",general)
