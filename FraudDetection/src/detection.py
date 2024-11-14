import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('fraud_data.csv')


# Function to refactor the date
def refactor_date(x):
    x = str(x).strip()  # Strip any extra spaces
    month, year = x.split("/")  # Split the date into month and year
    return int(month), int(year)

# Apply the function to the Expiry column
data[['Month_Expiry', 'Year_Expiry']] = data['Expiry'].apply(lambda x: pd.Series(refactor_date(x)))

# Apply the Profession transformation
data["Profession"] = data["Profession"].apply(lambda x: 1 if x == "DOCTOR" else 2 if x == "LAWYER" else 0)

# Drop the Expiry column
data.drop("Expiry", axis=1, inplace=True)

# Display the updated DataFrame
print(data)

X = data.drop(columns=["Fraud"]).values
y= data["Fraud"].values
logistic_regression = LogisticRegression()
logistic_regression.fit(X,y)
from sklearn.metrics import accuracy_score

predictions = logistic_regression.predict(X)
training_data_accuracy = accuracy_score(y, predictions)
print('Accuracy on Training data : ', training_data_accuracy)
data["Fraud_Predicted"] = predictions


data
print(data["Fraud_Predicted"].value_counts())

