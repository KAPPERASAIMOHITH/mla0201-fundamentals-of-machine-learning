import pandas as pd
# Sample data
data=pd.read_csv("CREDITSCORE.csv")
# Create a DataFrame
df = pd.DataFrame(data)
# Define attribute weights (you can adjust these based on importance)
weights = {
    "Age": 0.1,
    "Annual_Income": 0.3,
    "Monthly_Inhand_Salary": 0.2,
    "Num_Bank_Accounts": 0.05,
    "Num_Credit_Card": 0.05,
    "Interest_Rate": -0.1,  # Lower interest rate is better
    "Num_of_Loan": -0.1,  # Fewer loans are better
}
# Define a function to calculate credit score
def calculate_credit_score(row):
    score = 0
    for attribute, weight in weights.items():
        if attribute == "Type_of_Loan":
            # You can implement more complex logic here based on the types of loans
            pass
        else:
            score += row[attribute] * weight
    return score
# Apply the function to each row
df["Credit_Score"] = df.apply(calculate_credit_score, axis=1)

# Display the result
print(df[["ID", "Customer_ID", "Name", "Credit_Score"]])
