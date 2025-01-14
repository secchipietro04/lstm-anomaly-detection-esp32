import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
file_path = "./data_log_frigo.txt"  # Path to the first file
no_anomaly_path = "./data_log_ventilatoreB_V1_Still.txt"    # Path to the second file

# Define the column names
columns = ['Category', 'Timestamp', 'X', 'Y', 'Z']

# Read the CSV files into pandas DataFrames
df = pd.read_csv(file_path, names=columns)
#dfNA = pd.read_csv(no_anomaly_path, names=columns)

# Combine the DataFrames

# Remove the last 5 seconds of data
max_timestamp = df['Timestamp'].max()
df = df[df['Timestamp'] <= max_timestamp - 10]
#df = pd.concat([df, dfNA], ignore_index=True)

# Ignore the 'Timestamp' and use the row index as sequential time
df.reset_index(drop=True, inplace=True)

# Create separate plots for each category
categories = df['Category'].unique()
plt.figure(figsize=(12, 12))

for i, category in enumerate(categories, start=1):
    subset = df[df['Category'] == category]
    plt.subplot(len(categories), 1, i)
    plt.plot(subset.index, subset['X'], label='X')
    plt.plot(subset.index, subset['Y'], label='Y')
    plt.plot(subset.index, subset['Z'], label='Z')
    plt.title(f'Category {category}')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
