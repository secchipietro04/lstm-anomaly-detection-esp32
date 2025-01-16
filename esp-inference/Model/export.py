import re

# Path to your `model.h` file
header_file = "main/model.cc"
# Output `.tflite` file path
output_file = "model.tflite"

# Read the `model.h` file
with open(header_file, "r") as file:
    content = file.read()

# Regular expression to extract the byte array
array_pattern = r"const unsigned char g_model\[\] = \{([^}]+)\};"
match = re.search(array_pattern, content, re.DOTALL)

if not match:
    print("Could not find the byte array in the header file.")
    exit(1)

# Extract the bytes and clean them
byte_array = match.group(1).strip()
byte_values = re.findall(r"0x[0-9a-fA-F]{2}", byte_array)
byte_data = bytes(int(value, 16) for value in byte_values)

# Write the byte data to a `.tflite` file
with open(output_file, "wb") as file:
    file.write(byte_data)

print(f"Model has been successfully written to {output_file}")
