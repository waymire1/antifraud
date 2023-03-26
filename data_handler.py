import sqlite3
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn

class FraudDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FraudDetectionModel, self).__init__()

        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x



def create_database(database_name, column_data_types):
    # Connect to the database (this will create a new file if it doesn't exist)
    conn = sqlite3.connect(database_name)

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Define the table structure for storing transaction data
    columns_definition = ', '.join(f'{column} {data_type}' for column, data_type in column_data_types.items())
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS transactions (
            {columns_definition}
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def store_data_in_database(database_name, file_name, table_name):
    # Read the Excel file using pandas
    df = pd.read_excel(file_name)

    # Connect to the database
    conn = sqlite3.connect(database_name)

    # Store the data in the database
    df.to_sql(table_name, conn, if_exists='append', index=False)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def preprocess_data_for_pytorch(database_name):
    # Read the data from the database
    conn = sqlite3.connect(database_name)
    df = pd.read_sql_query('SELECT * FROM transactions', conn)
    conn.close()

    # Drop any rows with missing values
    df = df.dropna()

    # Convert categorical variables to numeric using LabelEncoder
    categorical_columns = ['payment_method', 'location']
    encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    # Select the features and target columns
    features = df[['amount', 'customer_id', 'payment_method', 'location', 'age']]
    target = df['fraudulent']

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Convert the data into PyTorch tensors
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    target_tensor = torch.tensor(target.values, dtype=torch.float32)

    return features_tensor, target_tensor
