import pickle
import numpy as np

# Path to your pickle file
path = "./pooled_meanstd.pk"  # or .pk if that's your file extension

# Try loading the file
with open(path, "rb") as f:
    data = pickle.load(f)

# Print type and optionally content
print("Type of loaded object:", type(data))

# If it's a dict, list, or tuple, you can inspect more:
if isinstance(data, dict):
    print("Keys:", data.keys())
    print(data)
elif isinstance(data, (list, tuple)):
    print("First few items:", data)
else:
    print("Content:", data)

# Check for missing values
print("\n" + "="*50)
print("CHECKING FOR MISSING VALUES")
print("="*50)

def check_missing_values(data, name="data"):
    """Helper function to check for missing values in different data types"""
    if isinstance(data, dict):
        print(f"Checking dictionary: {name}")
        for key, value in data.items():
            check_missing_values(value, f"{name}[{key}]")
    elif isinstance(data, (np.ndarray, list)):
        data_array = np.array(data)
        
        # Check for NaN values
        nan_count = np.isnan(data_array).sum()
        nan_percentage = (nan_count / data_array.size) * 100 if data_array.size > 0 else 0
        
        # Check for infinite values
        inf_count = np.isinf(data_array).sum()
        inf_percentage = (inf_count / data_array.size) * 100 if data_array.size > 0 else 0
        
        # Check for None values (if it's a list)
        none_count = 0
        if isinstance(data, list):
            none_count = sum(1 for item in data if item is None)
            none_percentage = (none_count / len(data)) * 100 if len(data) > 0 else 0
        else:
            none_percentage = 0
        
        print(f"  {name}:")
        print(f"    Shape: {data_array.shape}")
        print(f"    Total elements: {data_array.size}")
        print(f"    NaN values: {nan_count} ({nan_percentage:.2f}%)")
        print(f"    Inf values: {inf_count} ({inf_percentage:.2f}%)")
        if isinstance(data, list):
            print(f"    None values: {none_count} ({none_percentage:.2f}%)")
        
        # Check if there are any missing values
        has_missing = (nan_count > 0 or inf_count > 0 or none_count > 0)
        if has_missing:
            print(f"    ⚠️  CONTAINS MISSING VALUES!")
        else:
            print(f"    ✅ No missing values found")
        print()
        
        return has_missing
    else:
        print(f"  {name}: Type {type(data)} - cannot check for missing values")
        return False

# Check for missing values in the loaded data
has_missing = check_missing_values(data, "loaded_data")

# Calculate mean and std values (only if no missing values or handle them appropriately)
print("\n" + "="*50)
print("CALCULATING MEAN AND STD VALUES")
print("="*50)

if isinstance(data, dict):
    # If data is a dictionary, calculate for each key
    for key, value in data.items():
        if isinstance(value, (np.ndarray, list)):
            value_array = np.array(value)
            
            # Handle missing values for mean/std calculation
            if np.isnan(value_array).any() or np.isinf(value_array).any():
                print(f"Key: {key} - Contains missing values, using nanmean/nanstd")
                mean_val = np.nanmean(value_array)
                std_val = np.nanstd(value_array)
            else:
                mean_val = np.mean(value_array)
                std_val = np.std(value_array)
            
            print(f"Key: {key}")
            print(f"Mean: {mean_val:.6f}")
            print(f"Std:  {std_val:.6f}")
            print(f"Shape: {value_array.shape}")
            print()
elif isinstance(data, (np.ndarray, list)):
    # If data is a numpy array or list
    data_array = np.array(data)
    
    # Handle missing values for mean/std calculation
    if np.isnan(data_array).any() or np.isinf(data_array).any():
        print("Data contains missing values, using nanmean/nanstd")
        mean_val = np.nanmean(data_array)
        std_val = np.nanstd(data_array)
    else:
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
    
    print(f"Mean: {mean_val:.6f}")
    print(f"Std:  {std_val:.6f}")
    print(f"Shape: {data_array.shape}")
else:
    print("Data type not supported for mean/std calculation")

# Save the mean and std values to a new pickle file
print("\n" + "="*50)
print("SAVING MEAN AND STD VALUES")
print("="*50)

if isinstance(data, dict):
    # Create a new dictionary with mean and std for each key
    meanstd_dict = {}
    for key, value in data.items():
        if isinstance(value, (np.ndarray, list)):
            value_array = np.array(value)
            
            # Handle missing values for mean/std calculation
            if np.isnan(value_array).any() or np.isinf(value_array).any():
                mean_val = np.nanmean(value_array)
                std_val = np.nanstd(value_array)
            else:
                mean_val = np.mean(value_array)
                std_val = np.std(value_array)
            
            meanstd_dict[key] = {
                'mean': mean_val,
                'std': std_val,
                'shape': value_array.shape,
                'has_missing': np.isnan(value_array).any() or np.isinf(value_array).any()
            }
    
    # Save to a new pickle file
    # output_path = "../ssc/pm25_meanstd_calculated.pk"
    # with open(output_path, "wb") as f:
    #     pickle.dump(meanstd_dict, f)
    # print(f"Mean and std values saved to: {output_path}")
    print("Saved data structure:")
    for key, values in meanstd_dict.items():
        missing_flag = " (has missing values)" if values['has_missing'] else ""
        print(f"  {key}: mean={values['mean']:.6f}, std={values['std']:.6f}{missing_flag}")

elif isinstance(data, (np.ndarray, list)):
    # Save mean and std for single array
    data_array = np.array(data)
    
    # Handle missing values for mean/std calculation
    if np.isnan(data_array).any() or np.isinf(data_array).any():
        mean_val = np.nanmean(data_array)
        std_val = np.nanstd(data_array)
    else:
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
    
    meanstd_dict = {
        'mean': mean_val,
        'std': std_val,
        'shape': data_array.shape,
        'has_missing': np.isnan(data_array).any() or np.isinf(data_array).any()
    }
    
    # output_path = "../pm25/pm25_meanstd_calculated.pk"
    # with open(output_path, "wb") as f:
    #     pickle.dump(meanstd_dict, f)
    # print(f"Mean and std values saved to: {output_path}")
    missing_flag = " (has missing values)" if meanstd_dict['has_missing'] else ""
    print(f"Mean: {meanstd_dict['mean']:.6f}{missing_flag}")
    print(f"Std:  {meanstd_dict['std']:.6f}{missing_flag}")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
if has_missing:
    print("⚠️  The data contains missing values!")
    print("   - NaN values were handled using np.nanmean() and np.nanstd()")
    print("   - Consider data cleaning or imputation for better results")
else:
    print("✅ No missing values found in the data")
