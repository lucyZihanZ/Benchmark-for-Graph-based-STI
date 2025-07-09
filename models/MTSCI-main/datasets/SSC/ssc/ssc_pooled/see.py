import pickle

# Path to your pickle file
path = "./scaler.pkl"  # or .pk if that's your file extension

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
