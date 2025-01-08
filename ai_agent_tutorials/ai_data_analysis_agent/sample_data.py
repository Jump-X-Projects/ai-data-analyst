import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
base_date = datetime(2023, 1, 1)
dates = [base_date + timedelta(days=x) for x in range(200)]

# Create sample data
data = {
    'date': dates,
    'customer_id': np.random.randint(1000, 9999, 200),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], 200),
    'order_value': np.random.normal(500, 150, 200),  # Mean $500, std $150
    'items_purchased': np.random.randint(1, 10, 200),
    'google_ad_spend': np.random.normal(100, 30, 200),  # Mean $100, std $30
    'facebook_ad_spend': np.random.normal(80, 25, 200),  # Mean $80, std $25
    'customer_age': np.random.randint(18, 80, 200),
    'is_returning_customer': np.random.choice([True, False], 200),
    'shipping_method': np.random.choice(['Standard', 'Express', 'Next Day'], 200),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
}

df = pd.DataFrame(data)

# Add some calculated fields
df['total_ad_spend'] = df['google_ad_spend'] + df['facebook_ad_spend']
df['cost_per_item'] = df['order_value'] / df['items_purchased']

# Round numeric columns
df['order_value'] = df['order_value'].round(2)
df['google_ad_spend'] = df['google_ad_spend'].round(2)
df['facebook_ad_spend'] = df['facebook_ad_spend'].round(2)
df['total_ad_spend'] = df['total_ad_spend'].round(2)
df['cost_per_item'] = df['cost_per_item'].round(2)

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print("Sample data has been saved to sales_data.csv")