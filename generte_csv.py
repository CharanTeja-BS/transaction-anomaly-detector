import pandas as pd
import numpy as np
import os

# Create 'data' folder if not exists
if not os.path.exists('data'):
    os.makedirs('data')

# Number of transactions
num_records = 1000

# Random transaction data
np.random.seed(42)
data = {
    'transaction_amount': np.random.uniform(10, 5000, num_records),
    'transaction_type': np.random.choice(['payment', 'transfer', 'withdrawal', 'deposit'], num_records),
    'account_age_days': np.random.randint(30, 3000, num_records),
    'num_previous_transactions': np.random.randint(1, 500, num_records),
    'country': np.random.choice(['India', 'US', 'UK', 'Canada', 'Australia'], num_records),
    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], num_records),
    'is_foreign_transaction': np.random.choice([0, 1], num_records),
    'is_high_risk_country': np.random.choice([0, 1], num_records),
    'is_anomaly': np.random.choice([0, 1], num_records, p=[0.95, 0.05])  # 5% anomalies
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_path = 'data/transactions.csv'
df.to_csv(csv_path, index=False)

print(f"Synthetic transaction dataset generated at {csv_path}")
