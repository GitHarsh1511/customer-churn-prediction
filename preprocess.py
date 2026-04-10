import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Drop irrelevant columns
    df.drop(columns=['CustomerId', 'Surname'], inplace=True)

    # Encode categoricals
    le_gender    = LabelEncoder()
    le_location  = LabelEncoder()
    le_card      = LabelEncoder()

    df['Gender']    = le_gender.fit_transform(df['Gender'])
    df['Location']  = le_location.fit_transform(df['Location'])
    df['Card Type'] = le_card.fit_transform(df['Card Type'])

    # Features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and column names for use in Streamlit
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(X.columns), 'feature_columns.pkl')

    print("Scaler saved      : scaler.pkl")
    print("Columns saved     : feature_columns.pkl")

    return X_scaled, y, X.columns.tolist()


if __name__ == "__main__":
    X, y, cols = load_and_preprocess("data/Customer-Churn-Records.csv")
    print("\n--- Preprocessing Complete ---")
    print(f"Dataset shape  : {X.shape}")
    print(f"Churn rate     : {y.mean()*100:.2f}%")
    print(f"Features ({len(cols)}) : {cols}")