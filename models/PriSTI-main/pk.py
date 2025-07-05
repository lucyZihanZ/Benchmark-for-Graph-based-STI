import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
# Define the path to your dataset.
# IMPORTANT: Replace 'metr-la-features.csv' with the actual name of your CSV file
# that contains the node features (e.g., traffic speeds, sensor readings).
# Make sure this CSV file is in a 'data/metr_la/' directory relative to your script,
# or update this path to where your CSV file is located.
DATA_PATH = "./data/ssc/SSC_pooled.csv"
OUTPUT_DIR = "./data/ssc/"
OUTPUT_FILENAME = "ssc_meanstd.pk"

def create_meanstd_file(data_path, output_dir, output_filename):
    """
    Calculates the mean and standard deviation of a dataset from a CSV file
    and saves them to a pickle file.

    Args:
        data_path (str): The full path to the CSV dataset file (e.g., metr-la-features.csv).
        output_dir (str): The directory where the pickle file will be saved.
        output_filename (str): The name of the pickle file (e.g., metr_meanstd.pk).
    """
    print(f"Attempting to load data from: {data_path}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load the dataset from the CSV file
        # You might need to add other parameters for pd.read_csv
        # like 'sep' for delimiter, 'header' if no header row, etc.
        # For example: df = pd.read_csv(data_path, sep=',')
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data. Shape: {df.shape}")
        print("First 5 rows of the dataset:")
        print(df.head())
        df = df.select_dtypes(include=np.number)

        # Convert the DataFrame to a NumPy array for calculation
        # This assumes you want to calculate mean/std across all numerical columns.
        # If your CSV has non-numerical columns (e.g., timestamps, IDs), you MUST
        # select only the numerical columns before converting to numpy array.
        # Example: data = df[['feature_col1', 'feature_col2']].values
        data = df.values

        # Calculate mean and standard deviation
        # axis=0 means calculate across columns (for each feature)
        # keepdims=True ensures the output shape is (1, num_features) which is
        # often convenient for broadcasting during normalization.
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)

        # Handle cases where standard deviation might be zero (e.g., constant features)
        # Adding a small epsilon prevents division by zero during normalization.
        std[std == 0] = 1e-5 # A small value to avoid division by zero

        print(f"Calculated Mean (first 5 values): {mean.flatten()[:5]}")
        print(f"Calculated Std (first 5 values): {std.flatten()[:5]}")

        # Create a dictionary to store mean and std
        data_to_save = {
            'mean': mean,
            'std': std
        }

        # Define the full path for the output pickle file
        output_filepath = os.path.join(output_dir, output_filename)

        # Save the dictionary to a pickle file
        with open(output_filepath, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"Successfully created '{output_filename}' at '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{data_path}'.")
        print("Please ensure the CSV file exists at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Create dummy data for demonstration if the real CSV file doesn't exist
    # In a real scenario, you would have your metr-la-features.csv file.
    if not os.path.exists(DATA_PATH):
        print(f"'{DATA_PATH}' not found. Creating a dummy CSV file for demonstration.")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        dummy_data = pd.DataFrame(np.random.rand(100, 10) * 100, columns=[f'feature_{i}' for i in range(10)])
        dummy_data.to_csv(DATA_PATH, index=False) # index=False prevents writing the DataFrame index as a column
        print("Dummy 'metr-la-features.csv' created.")

    create_meanstd_file(DATA_PATH, OUTPUT_DIR, OUTPUT_FILENAME)

    # Optional: Verify the content of the created .pk file
    try:
        with open(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), 'rb') as f:
            loaded_data = pickle.load(f)
            print("\n--- Verifying loaded data ---")
            print(f"Loaded Mean (first 5 values): {loaded_data['mean'].flatten()[:5]}")
            print(f"Loaded Std (first 5 values): {loaded_data['std'].flatten()[:5]}")
            print("Verification successful.")
    except FileNotFoundError:
        print(f"Error: Could not verify '{OUTPUT_FILENAME}', file not found after creation attempt.")
    except Exception as e:
        print(f"Error during verification: {e}")