
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
# Example DataFrame df
# df = pd.DataFrame(...)
df = pd.read_csv('fraud_data.csv')
df['Exp_Month'] = df['Expiry'].str.split('/').str[0]
df['Exp_Year'] = df['Expiry'].str.split('/').str[1]
df.drop(columns =['Credit_card_number','Expiry'],inplace = True)
df['Exp_Month'] = df['Exp_Month'].astype(int)
df['Exp_Year'] = df['Exp_Year'].astype(int)


# Define your categorical and numerical columns
categorical_cols = ['Profession']  # Replace with actual categorical column names
numerical_cols = ['Income', 'Security_code','Exp_Month','Exp_Year']  # Replace with actual numerical column names

# Check if the columns exist in the DataFrame
missing_categorical_cols = [col for col in categorical_cols if col not in df.columns]
missing_numerical_cols = [col for col in numerical_cols if col not in df.columns]

if missing_categorical_cols:
    print(f"Missing categorical columns: {missing_categorical_cols}")
if missing_numerical_cols:
    print(f"Missing numerical columns: {missing_numerical_cols}")

# If there are no missing columns, proceed with the splitting
if not missing_categorical_cols and not missing_numerical_cols:
    # Split the data into features (X) and target (y)
    X = df.drop('Fraud', axis=1)  # 'Fraud' is the target column
    y = df['Fraud']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the preprocessing pipeline for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # Standardize numerical columns
            ('cat', OrdinalEncoder(), categorical_cols)  # Apply ordinal encoding to categorical columns
        ]
    )

    # Create a pipeline with SMOTE, preprocessing, and Logistic Regression
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # Apply SMOTE to balance the classes
        ('classifier', LogisticRegression(random_state=42))  # Logistic Regression model
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model (you can use accuracy, classification report, confusion matrix, etc.)
    print(classification_report(y_test, y_pred))
else:
    print("Cannot proceed with model training due to missing columns.")
