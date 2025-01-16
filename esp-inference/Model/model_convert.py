import argparse

def export_model_to_cc(input_file, output_file):
    with open(input_file, "rb") as f:
        model_content = f.read()

    with open(output_file, "w") as f:
        f.write('#include "model.h"\n\n')
        f.write('alignas(8) const unsigned char g_model[] = {\n')

        # Write the model data as hexadecimal values
        for i, byte in enumerate(model_content):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:  # 12 bytes per line
                f.write("\n")
        
        f.write('\n};\n')
        f.write(f"const int g_model_len = {len(model_content)};\n")

def main():
    parser = argparse.ArgumentParser(description="Export TFLite model to C++ source file.")
    parser.add_argument("input_file", help="Path to the input TFLite model file")
    parser.add_argument("output_file", help="Path to the output C++ source file")
    args = parser.parse_args()

    export_model_to_cc(args.input_file, args.output_file)
    print(f"Model exported to {args.output_file}")

if __name__ == "__main__":
    main()
