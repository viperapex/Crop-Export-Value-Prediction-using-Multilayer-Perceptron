import pandas as pd
import glob
import os

# Specify the file path of the CSV file you want to load
folder_path = r"content\data\\"  # Use raw string and ensure the path is correct

# Get a list of all CSV files in the folder
csv_files = glob.glob(folder_path + "*.csv")
print("Found CSV files:", csv_files)  # Check which files are being found

# Create an empty list to store DataFrames
data_frames = []

# Read each CSV file and append its DataFrame to the list
for file in csv_files:
    print("Reading file:", file)  # See which file is being read
    df = pd.read_csv(file)
    print("Loaded DataFrame size:", df.shape)  # Check the size of DataFrame
    data_frames.append(df)

# Concatenate all DataFrames into a single DataFrame, if any DataFrames were loaded
if data_frames:
    data = pd.concat(data_frames, ignore_index=True)
    # Display the first few rows of the concatenated DataFrame
    print(data.head())
else:
    print("No CSV files were loaded into DataFrames.")






# Create an empty list to store DataFrames
data_frames = []

# Read each CSV file and append its DataFrame to the list
for file in csv_files:
    df = pd.read_csv(file)
    data_frames.append(df)

# Concatenate all DataFrames into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Display the first few rows of the concatenated DataFrame
print(data.head())

# List all files in the folder
files = os.listdir(folder_path)

# Filter CSV files
csv_files = [file for file in files if file.endswith('.csv')]

# Print the names of CSV files
for file in csv_files:
    print(file)

# Create an empty list to store DataFrames
data_frames = []

# Read each CSV file and append its DataFrame to the list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    data_frames.append(df)

# Concatenate all DataFrames into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Display the first few rows of the concatenated DataFrame
print(data.head())

# Loop through each CSV file and count columns
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    num_columns = len(df.columns)
    print(f"Number of columns in '{file}': {num_columns}")

# Loop through each CSV file and print column names and count
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    num_columns = len(df.columns)
    print(f"Number of columns in '{file}': {num_columns}")
    print(f"Column names: {list(df.columns)}\n")





# Check for missing values in the data
missing_values = data.isnull().sum()

# Display columns with missing values
print("Columns with missing values:")
print(missing_values[missing_values > 0])






# Check for missing values in the data
missing_values = data.isnull().sum()

# Display columns with missing values
print("Columns with missing values:")
print(missing_values[missing_values > 0])






# Handle missing values
data.dropna(inplace=True)  # Drop rows with missing values




# Check the data types of each column
data_types = data.dtypes

# Select columns with categorical data types
categorical_columns = data_types[data_types == 'object'].index.tolist()

print("Categorical columns:")
print(categorical_columns)







# One-hot encode categorical features
encoded_data = pd.get_dummies(data, columns=categorical_columns)

# Display the first few rows of the encoded DataFrame
print(encoded_data.head())



# One-hot encode categorical features
encoded_data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)



# Review the data types of each column
print(data.dtypes)


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data from all CSV files
folder_path = r"C:\Users\Prahlad\OneDrive - University of Sussex\Study\AI and Adaptive Systems\Machine Learning\ML-Assignment\content\data\\"
csv_files = ["Emissions - FAOSTAT_data_en_2-27-2024.csv", "Employment - FAOSTAT_data_en_2-27-2024.csv",
             "Exchange rate - FAOSTAT_data_en_2-22-2024.csv", "Crops production indicators - FAOSTAT_data_en_2-22-2024.csv",
             "Fertilizers use - FAOSTAT_data_en_2-27-2024.csv", "Consumer prices indicators - FAOSTAT_data_en_2-22-2024.csv",
             "Food trade indicators - FAOSTAT_data_en_2-22-2024.csv", "Foreign direct investment - FAOSTAT_data_en_2-27-2024.csv",
             "Food balances indicators - FAOSTAT_data_en_2-22-2024.csv", "Land temperature change - FAOSTAT_data_en_2-27-2024.csv",
             "Pesticides use - FAOSTAT_data_en_2-27-2024.csv", "Land use - FAOSTAT_data_en_2-22-2024.csv",
             "Food security indicators  - FAOSTAT_data_en_2-22-2024.csv"]

data_frames = []
for file in csv_files:
    file_path = folder_path + file
    df = pd.read_csv(file_path)
    data_frames.append(df)

# Concatenate all DataFrames into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Preprocessing
# Review data types
print("Data Types:\n", data.dtypes)

# Identify numerical columns
numerical_columns = data.select_dtypes(include=['number']).columns
if numerical_columns.empty:
    raise ValueError("No numerical columns found in the data.")

# Identify categorical columns
categorical_columns = [col for col in data.columns if col not in numerical_columns]
if not categorical_columns:
    raise ValueError("No categorical columns found in the data.")

# One-hot encode categorical features
encoded_data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Identify encoded categorical columns
encoded_categorical_columns = [col for col in encoded_data.columns if col not in numerical_columns]
if not encoded_categorical_columns:
    raise ValueError("No categorical columns found in the encoded data.")

# Initialize StandardScaler
scaler = StandardScaler()

# Scale numerical features
scaled_numerical_data = scaler.fit_transform(encoded_data[numerical_columns])

# Create DataFrame for scaled numerical features
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_columns)

# Combine scaled numerical features with encoded categorical features
preprocessed_data = pd.concat([scaled_numerical_df, encoded_data.drop(columns=numerical_columns)], axis=1)

# Display the first few rows of the preprocessed data
print("Preprocessed Data:\n", preprocessed_data.head())


# Display the first few rows of the dataset
print(data.head())

# Review the data types of each column
print(data.dtypes)

# Print the column names of the DataFrame
print(data.columns)


# Define the categories of features to include
crop_production_features = ['Item', 'Year', 'Value']  # Assuming 'Item' refers to different crops
economic_indicators = ['Item', 'Year', 'Value']  # Assuming economic indicators have 'Item' and 'Year' columns
climate_environmental_factors = ['Item', 'Year', 'Value']  # Assuming climate factors have 'Item' and 'Year' columns
market_trade_indicators = ['Item', 'Year', 'Value']  # Assuming market trade indicators have 'Item' and 'Year' columns
demographic_data = ['Item', 'Year', 'Value']  # Assuming demographic data has 'Item' and 'Year' columns

# Combine all feature categories into a single list
selected_features = (crop_production_features + economic_indicators + climate_environmental_factors +
                     market_trade_indicators + demographic_data)

# Select the features from the dataset
selected_data = data[selected_features]

# Display the selected features
print(selected_data.head())


# Define the categories of features to include
crop_production_features = ['Item', 'Year', 'Value']  # Assuming 'Item' refers to different crops
economic_indicators = ['Item', 'Year', 'Value']  # Assuming economic indicators have 'Item' and 'Year' columns
climate_environmental_factors = ['Item', 'Year', 'Value']  # Assuming climate factors have 'Item' and 'Year' columns
market_trade_indicators = ['Item', 'Year', 'Value']  # Assuming market trade indicators have 'Item' and 'Year' columns
demographic_data = ['Item', 'Year', 'Value']  # Assuming demographic data has 'Item' and 'Year' columns

# Combine all feature categories into a single list
selected_features = (crop_production_features + economic_indicators + climate_environmental_factors +
                     market_trade_indicators + demographic_data)




# Rename columns to remove spaces and special characters
selected_data.columns = selected_data.columns.str.replace(' ', '_')  # Replace spaces with underscores
selected_data.columns = selected_data.columns.str.replace('[^a-zA-Z0-9_]', '')  # Remove special characters

print(type(selected_data['Year']))


print(selected_data['Year'].iloc[:, 0].unique())



from sklearn.model_selection import train_test_split

# Assuming 'preprocessed_data' is your DataFrame containing all the preprocessed data
# Split the data into features and target variable (assuming 'target' as the column name for target variable)
X = preprocessed_data.drop('Value', axis=1)  # Features
y = preprocessed_data['Value']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shapes of the resulting data sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)



from sklearn.neural_network import MLPClassifier  # Use MLPRegressor if your task is regression

# Define the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100,),  # one hidden layer with 100 neurons
                    activation='relu',  # using ReLU activation function
                    solver='adam',  # default solver that works well in practice
                    max_iter=200,  # maximum number of iterations to convergence
                    random_state=42,  # for reproducibility
                    learning_rate_init=0.001)  # starting learning rate


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
import sys
from tqdm import tqdm

# Custom MLPRegressor to show tqdm progress bar
class MLPRegressorWithProgress(MLPRegressor):
    def _fit_stochastic(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental):
        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)
        with tqdm(total=self.max_iter, desc="Training progress", file=sys.stdout, dynamic_ncols=True) as progress_bar:
            for i in super()._fit_stochastic(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental):
                progress_bar.update(1)
                yield i

# Imputer for features
feature_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train_imputed = feature_imputer.fit_transform(X_train)
X_test_imputed = feature_imputer.transform(X_test)

# Imputer for the target variable
target_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
y_train_imputed = target_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_imputed = target_imputer.transform(y_test.values.reshape(-1, 1)).ravel()

# Initialize and fit the MLPRegressor with progress bar
mlp_regressor = MLPRegressorWithProgress(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200, random_state=42, verbose=False)
mlp_regressor.fit(X_train_imputed, y_train_imputed)




