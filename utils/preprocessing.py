import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load CSV data into pandas DataFrame"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Separate features and target
    X = df.drop("is_anomaly", axis=1)
    y = df["is_anomaly"]

    # One-hot encode any categorical columns
    X = pd.get_dummies(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the numeric data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
