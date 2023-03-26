#This program is an ANTIfraud detection program that uses pytorch to create a simple forward passing neural network. 
#The program is a GUI that allows the user to create a database, crawl data from an excel file, and train a model. 
#The program also allows the user to save and load the model. 
#The program is written in python and uses pyqt5 for the GUI. 
#The program is a work in progress and is not complete.
#Author = "Rex Waymire"
#Date = "3/25/2023"
#Version = "1.0"
#Free use.  Just remember that you have to trian the model to calibrate the program to work, so the longer you build the dataset the more accurate it is. 


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import torch
from PyQt5.QtWidgets import QTableWidge
from PyQt5.QtWidgets import QTableWidgetItem, QAbstractItemView
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def save_model(self):
    if not hasattr(self, 'model'):
        self.status_label.setText("No model available to save")
        return

    model_path = "fraud_detection_model.pt"
    torch.save(self.model.state_dict(), model_path)
    self.status_label.setText(f'Model saved to {model_path}')

def load_model(self):
    model_path = "fraud_detection_model.pt"

    if not os.path.exists(model_path):
        self.status_label.setText(f"Model file not found: {model_path}")
        return

    if not hasattr(self, 'model'):
        self.init_model()

    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()
    self.status_label.setText(f'Model loaded from {model_path}')


class AntiFraudApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface
        self.init_ui()

    def preprocess_data(self, df):
        # Drop rows with missing values
        df = df.dropna()

        # TODO: Add any additional preprocessing steps based on your specific requirements
        return df

    def create_database(self):
        column_data_types = {
            'id': 'INTEGER PRIMARY KEY',
            'date': 'TEXT',
            'category': 'TEXT',
            'amount': 'REAL',
            'customer_id': 'INTEGER',
            'payment_method': 'TEXT',
            'location': 'TEXT',
            'age': 'INTEGER',
            'fraudulent': 'INTEGER'
        }
        database_name = 'transactions.db'
        create_database(database_name, column_data_types)
        self.status_label.setText(f'Database created: {database_name}')

    def crawl_and_store_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Excel File', '',
                                                    'Excel Files (*.xlsx);;All Files (*)',
                                                    options=options)

        if file_name:
            self.status_label.setText(f'Crawling data from: {file_name}')
            store_data_in_database('transactions.db', file_name, 'transactions')
            self.status_label.setText(f'Data crawled and stored from: {file_name}')

        #pytorch stuff
    def train_model(self):
        # Load the data from the SQLite database
        sqlite3.connect(database_name)
        conn = sqlite3.connect(database_name)
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
        conn.close()

        # Preprocess the data and convert it into PyTorch tensors
        features = df[['amount', 'customer_id', 'payment_method', 'location', 'age']].values
        labels = df['fraudulent'].values

        # Scale the features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Convert the data into PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Create DataLoader objects for the training and validation sets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Define a loss function and an optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model in batches for a certain number of epochs
        num_epochs = 10
        for epoch in range(num_epochs):
         
        # Train the model
            self.model.train()
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validate the model
            self.model.eval()
            val_loss = 0.0
            for batch in val_loader:
                inputs, targets = batch
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)

            # Print the progress
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

  
    def get_data_tensors(self):
        features_tensor, target_tensor = preprocess_data_for_pytorch('transactions.db')
        print("Features tensor:", features_tensor)
        print("Target tensor:", target_tensor)


    def label_fraud(self):
        selected_row = self.transactions_table.currentRow()

        if selected_row == -1:
            self.status_label.setText("No transaction selected")
            return

        selected_transaction_id = self.transactions_table.item(selected_row, 0).text()

        conn = sqlite3.connect('transactions.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE transactions SET fraudulent = 1 WHERE id = ?', (selected_transaction_id,))
        conn.commit()
        conn.close()

        self.status_label.setText(f"Labeled transaction {selected_transaction_id} as fraudulent")

    def init_model(self):
        input_size = 5  # The number of features in your dataset
        hidden_size = 32  # The number of units in the hidden layer
        output_size = 1  # The number of output units (1 for binary classification)

        self.model = FraudDetectionModel(input_size, hidden_size, output_size)


    def run_pytorch_analysis(self):
        database_name = 'transactions.db'
        table_name = 'transactions'
        self.train_model()
        self.status_label.setText(f'PyTorch analysis completed')

        # Add other plot functions here
    def plot_scatter_plot(self, df, x_column, y_column):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=x_column, y=y_column, data=df)
        plt.title(f'{y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def plot_payment_method_transactions(self, df, payment_method_column):
        payment_method_transactions = df[payment_method_column].value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=payment_method_transactions.index, y=payment_method_transactions.values)
        plt.title('Number of Transactions per Payment Method')
        plt.xlabel('Payment Method')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

    def plot_location_transactions(self, df, location_column):
        location_transactions = df[location_column].value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=location_transactions.index, y=location_transactions.values)
        plt.title('Number of Transactions per Location')
        plt.xlabel('Location')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

    def plot_time_series(self, df, date_column, amount_column):
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_column], df[amount_column])
        plt.title('Transaction Amounts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Transaction Amount')
        plt.show()

    def plot_box_plot(self, df, category_column, amount_column):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=category_column, y=amount_column, data=df)
        plt.title('Transaction Amounts per Category')
        plt.xlabel('Category')
        plt.ylabel('Transaction Amount')
        plt.xticks(rotation=45)
        plt.show()

    def plot_heatmap(self, df, date_column):
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['hour'] = df[date_column].dt.hour
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack()

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d')
        plt.title('Transactions per Day and Hour')
        plt.xlabel('Hour')
        plt.ylabel('Day of Week')
        plt.show()

    def plot_customer_transactions(self, df, customer_column):
        customer_transactions = df[customer_column].value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=customer_transactions.index, y=customer_transactions.values)
        plt.title('Number of Transactions per Customer')
        plt.xlabel('Customer ID')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

    def process_file(self, file_name):
        # Read the Excel file using pandas
        df = pd.read_excel(file_name)

        # Data preprocessing
        df = self.preprocess_data(df)

        self.plot_time_series(df, 'date', 'amount')
        self.plot_box_plot(df, 'category', 'amount')
        self.plot_heatmap(df, 'date')
        self.plot_scatter_plot(df, 'age', 'amount')
        self.plot_customer_transactions(df, 'customer_id')
        self.plot_payment_method_transactions(df, 'payment_method')
        self.plot_location_transactions(df, 'location')

    def init_ui(self):
        # Set the main window properties
        self.setWindowTitle("Financial Anti-Fraud Analysis")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget to hold the layout and widgets
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Set up the main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add widgets to the layout
        load_button = QPushButton('Load Excel File')
        load_button.clicked.connect(self.load_excel_file)
        main_layout.addWidget(load_button)

        # Add a button for creating the database
        create_database_button = QPushButton('Create Database')
        create_database_button.clicked.connect(self.create_database)
        main_layout.addWidget(create_database_button)

        # Add a button for crawling and storing data
        crawl_data_button = QPushButton('Crawl and Store Data')
        crawl_data_button.clicked.connect(self.crawl_and_store_data)
        main_layout.addWidget(crawl_data_button)

        # Run PyTorch Analysis
        pytorch_analysis_button = QPushButton('Run PyTorch Analysis')
        pytorch_analysis_button.clicked.connect(self.run_pytorch_analysis)
        main_layout.addWidget(pytorch_analysis_button)
        
        # Add a button for getting the data tensors
        get_data_tensors_button = QPushButton('Get Data Tensors')
        get_data_tensors_button.clicked.connect(self.get_data_tensors)
        main_layout.addWidget(get_data_tensors_button)

        # Add a button for training the model
        train_model_button = QPushButton('Train Model')
        train_model_button.clicked.connect(self.train_model)
        main_layout.addWidget(train_model_button)

        # Add a button for saving the model
        save_model_button = QPushButton('Save Model')
        save_model_button.clicked.connect(self.save_model)
        main_layout.addWidget(save_model_button)

        # Add a button for loading the model
        load_model_button = QPushButton('Load Model')
        load_model_button.clicked.connect(self.load_model)
        main_layout.addWidget(load_model_button)


        #labal fraud button
        label_fraud_button = QPushButton('Label Selected Transaction as Fraudulent')
        label_fraud_button.clicked.connect(self.label_fraud)
        main_layout.addWidget(label_fraud_button)
        
        ##transaction table
        self.transactions_table = QTableWidget()
        main_layout.addWidget(self.transactions_table)


        # Add a label to display the status of the application
        self.status_label = QLabel('No file loaded')
        main_layout.addWidget(self.status_label)

        # Add a button for initializing the model
        init_model_button = QPushButton('Initialize Model')
        init_model_button.clicked.connect(self.init_model)
        main_layout.addWidget(init_model_button)


    def load_excel_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Excel File', '', 'Excel Files (*.xlsx);;All Files (*)', options=options)

        if file_name:
            self.status_label.setText(f'Loaded file: {file_name}')
            self.process_file(file_name)
            self.display_transactions_in_table()  # Add this line

    def display_transactions_in_table(self):
        conn = sqlite3.connect('transactions.db')
        df = pd.read_sql_query('SELECT * FROM transactions', conn)
        conn.close()

        self.transactions_table.setRowCount(df.shape[0])
        self.transactions_table.setColumnCount(df.shape[1])
        self.transactions_table.setHorizontalHeaderLabels(df.columns)
        self.transactions_table.setSelectionBehavior(QAbstractItemView.SelectRows)  # Allow selecting entire rows

        for i, row in df.iterrows():
            for j, cell in enumerate(row):
                self.transactions_table.setItem(i, j, QTableWidgetItem(str(cell)))

        self.transactions_table.resizeColumnsToContents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AntiFraudApp()
    main_window.show()
    sys.exit(app.exec_())
