import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
def read_and_separate_sensor_data_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['Category', 'Timestamp', 'X', 'Y', 'Z'])
    
    lowest_timestamp = df['Timestamp'].min()
    highest_timestamp = df['Timestamp'].max()
    # Ensure data types are properly set
    df['Category'] = df['Category'].astype(int)
    #df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  # Assuming the timestamp is in seconds
    df['X'] = df['X'].astype(float)
    df['Y'] = df['Y'].astype(float)
    df['Z'] = df['Z'].astype(float)
    
    # Separate the DataFrame into three based on the Category
    gyro_df = df[df['Category'] == 0]
    acceleration_df = df[df['Category'] == 1]
    linear_acceleration_df = df[df['Category'] == 2]
    
    return gyro_df, acceleration_df, linear_acceleration_df, lowest_timestamp, highest_timestamp
def interpolate_with_spline(df, target_num_rows):
    # Get the original timestamps and data columns
    old_indices = df['Timestamp'].values
    new_length = target_num_rows
    new_indices = np.linspace(old_indices.min(), old_indices.max(), new_length)

    # Initialize a new DataFrame for the interpolated data
    interpolated_df = pd.DataFrame({'Timestamp': new_indices})

    # Interpolate each data column separately
    for col in ['Category', 'X', 'Y', 'Z']:
        if col in df.columns:
            # Ensure there are no NaNs in the original data before fitting the spline
            mask = ~np.isnan(df[col].values)
            spl = UnivariateSpline(old_indices[mask], df[col].values[mask], k=3, s=0)
            interpolated_df[col] = spl(new_indices)

    return interpolated_df
def plot_data(original_df, interpolated_df, column_name):
    plt.figure(figsize=(12, 6))
    plt.plot(original_df['Timestamp'], original_df[column_name], '-', label='Original Data', markersize=4)
    plt.plot(interpolated_df['Timestamp'], interpolated_df[column_name], '-', label='Interpolated Data', linewidth=1)
    plt.title(f'Interpolation of {column_name}')
    plt.xlabel('Timestamp')
    plt.ylabel(column_name)
    plt.legend()
    plt.show()


target_num_rows = 15400
gyro_df, acceleration_df, linear_acceleration_df, lowest_timestamp, highest_timestamp = read_and_separate_sensor_data_csv('data_log_frigo.txt')

print(interpolate_with_spline(gyro_df, target_num_rows))

plot_data(gyro_df, interpolate_with_spline(gyro_df, target_num_rows), 'X')