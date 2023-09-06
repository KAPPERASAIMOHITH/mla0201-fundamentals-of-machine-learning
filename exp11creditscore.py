import pandas as pd
# Sample data
data=pd.read_csv("CREDITSCORE.csv")
# Create a DataFrame
df = pd.DataFrame(data)
weights = {
    "Age": 0.1,
    "Annual_Income": 0.3,
    "Monthly_Inhand_Salary": 0.2,
    "Num_Bank_Accounts": 0.05,
    "Num_Credit_Card": 0.05,
    "Interest_Rate": -0.1,  # Lower interest rate is better
    "Num_of_Loan": -0.1,  # Fewer loans are better
}
def calculate_credit_score(row):
    score = 0
    for attribute, weight in weights.items():
        if attribute == "Type_of_Loan":
            pass
        else:
            score += row[attribute] * weight
    return score
df["Credit_Score"] = df.apply(calculate_credit_score, axis=1)
print(df[["ID", "Customer_ID", "Name", "Credit_Score"]])
