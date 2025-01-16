import tensorflow as tf

# Path to the `.tflite` file
model_path = "model.tflite"

# Load the model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model details
print("=== Model Information ===")
print(f"Number of Inputs: {len(input_details)}")
print(f"Number of Outputs: {len(output_details)}")
print("")

# Input details
print("=== Input Details ===")
for i, input_detail in enumerate(input_details):
    print(f"Input {i}:")
    print(f"  Name: {input_detail['name']}")
    print(f"  Shape: {input_detail['shape']}")
    print(f"  Dtype: {input_detail['dtype']}")
    print("")

# Output details
print("=== Output Details ===")
for i, output_detail in enumerate(output_details):
    print(f"Output {i}:")
    print(f"  Name: {output_detail['name']}")
    print(f"  Shape: {output_detail['shape']}")
    print(f"  Dtype: {output_detail['dtype']}")
    print("")

# Get all tensors
tensor_details = interpreter.get_tensor_details()

print("=== Tensor Details ===")
for tensor in tensor_details:
    print(f"Tensor Name: {tensor['name']}")
    print(f"Index: {tensor['index']}")
    print(f"Shape: {tensor['shape']}")
    print(f"Type: {tensor['dtype']}")
    print("")
