/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ==============================================================================*/

#include "main_functions.h"

#include "constants.h"
#include "esp_system.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SEQUENCE_LENGTH 10
#define LSTM_UNITS 16
#define FEATURE_SIZE 3

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 64000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setupModel() {
    // Map the model into a usable data structure.
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf(
            "Model provided is schema version %d not equal to supported "
            "version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Pull in only the operation implementations we need.
    static tflite::MicroMutableOpResolver<26> resolver;
    if (resolver.AddUnidirectionalSequenceLSTM() != kTfLiteOk) {
        return;
    }
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        return;
    }
    if (resolver.AddReshape() != kTfLiteOk) {
        return;
    }
    if (resolver.AddWhile() != kTfLiteOk) {
        return;
    }
    if (resolver.AddLess() != kTfLiteOk) {
        return;
    }
    if (resolver.AddLogicalAnd() != kTfLiteOk) {
        return;
    }
    if (resolver.AddAdd() != kTfLiteOk) {
        return;
    }
    if (resolver.AddGather() != kTfLiteOk) {
        return;
    }
    if (resolver.AddSplit() != kTfLiteOk) {
        return;
    }
    if (resolver.AddLogistic() != kTfLiteOk) {
        return;
    }
    if (resolver.AddTranspose() != kTfLiteOk) {
        return;
    }
    if (resolver.AddStridedSlice() != kTfLiteOk) {
        return;
    }
    if (resolver.AddFill() != kTfLiteOk) {
        return;
    }
    if (resolver.AddExpandDims() != kTfLiteOk) {
        return;
    }
    if (resolver.AddConcatenation() != kTfLiteOk) {
        return;
    }
    if (resolver.AddMul() != kTfLiteOk) {
        return;
    }
    if (resolver.AddTanh() != kTfLiteOk) {
        return;
    }
    if (resolver.AddSlice() != kTfLiteOk) {
        return;
    }
    if (resolver.AddShape() != kTfLiteOk) {
        return;
    }
    if (resolver.AddUnpack() != kTfLiteOk) {
        return;
    }
    if (resolver.AddPack() != kTfLiteOk) {
        return;
    }

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Add this after interpreter creation
    for (int i = 0; i < interpreter->inputs_size(); i++) {
        TfLiteTensor* tensor = interpreter->input(i);
        MicroPrintf("Input %d: dims=%d shape=%d,%d,%d type=%d", i,
                    tensor->dims->size, tensor->dims->data[0],
                    tensor->dims->data[1], tensor->dims->data[2], tensor->type);
    }
}

// Helper function to print tensor details
void printTensorDetails(const TfLiteTensor* tensor, const char* name) {
    MicroPrintf("Tensor %s details:", name);
    MicroPrintf("- dims->size: %d", tensor->dims->size);
    for (int i = 0; i < tensor->dims->size; i++) {
        MicroPrintf("- dims->data[%d]: %d", i, tensor->dims->data[i]);
    }
    MicroPrintf("- type: %d", tensor->type);
}

void printAllTensorShapes() {
    for (int i = 0; i < 5; i++) {
        TfLiteTensor* tensor = interpreter->input(i);
        MicroPrintf("Input tensor %d:", i);
        MicroPrintf("- dims->size: %d", tensor->dims->size);
        for (int j = 0; j < tensor->dims->size; j++) {
            MicroPrintf("- dims->data[%d]: %d", j, tensor->dims->data[j]);
        }
    }
}

// The name of this function is important for Arduino compatibility.
void setup() {
    setupModel();

    // Initialize other variables and state here

    // Debug print all tensor shapes
    printAllTensorShapes();
}

bool isValidFloat(float x) { return !std::isnan(x) && !std::isinf(x); }

void loop() {
    // States for first LSTM layer
    static float hidden_state_1[16] = {0};  // [1,16] from first LSTM layer
    static float cell_state_1[16] = {0};    // [1,16] from first LSTM layer

    // States for second LSTM layer
    static float hidden_state_2[8] = {0};  // [1,8] from second LSTM layer
    static float cell_state_2[8] = {0};    // [1,8] from second LSTM layer

    static float input_buffer[3] = {0};   // [1,3] input features
    static float output_sequence[10][3];  // Store outputs

    // Reset all states
    memset(hidden_state_1, 0, sizeof(hidden_state_1));
    memset(cell_state_1, 0, sizeof(cell_state_1));
    memset(hidden_state_2, 0, sizeof(hidden_state_2));
    memset(cell_state_2, 0, sizeof(cell_state_2));

    // Process each step in the sequence
    for (int step = 0; step < SEQUENCE_LENGTH; step++) {
        // Generate input data (replace with your sensor data)
        for (int i = 0; i < 3; i++) {
            input_buffer[i] = ((float)rand() / RAND_MAX);
        }

        // Debug print inputs
        MicroPrintf("Step %d Input: %.3f %.3f %.3f", step, input_buffer[0],
                    input_buffer[1], input_buffer[2]);

        // Set up model input tensors
        TfLiteTensor* input_tensor = interpreter->input(0);
        TfLiteTensor* hidden_state_tensor_1 = interpreter->input(1);
        TfLiteTensor* cell_state_tensor_1 = interpreter->input(2);
        TfLiteTensor* hidden_state_tensor_2 = interpreter->input(3);
        TfLiteTensor* cell_state_tensor_2 = interpreter->input(4);

        // Copy data to input tensors
        memcpy(input_tensor->data.f, input_buffer, 3 * sizeof(float));
        memcpy(hidden_state_tensor_1->data.f, hidden_state_1,
               16 * sizeof(float));
        memcpy(cell_state_tensor_1->data.f, cell_state_1, 16 * sizeof(float));
        memcpy(hidden_state_tensor_2->data.f, hidden_state_2,
               8 * sizeof(float));
        memcpy(cell_state_tensor_2->data.f, cell_state_2, 8 * sizeof(float));

        // Run inference
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            MicroPrintf("Invoke failed at step %d", step);
            return;
        }

        // Get output tensors
        TfLiteTensor* output_tensor = interpreter->output(0);
        TfLiteTensor* new_hidden_state_tensor_1 = interpreter->output(1);
        TfLiteTensor* new_cell_state_tensor_1 = interpreter->output(2);
        TfLiteTensor* new_hidden_state_tensor_2 = interpreter->output(3);
        TfLiteTensor* new_cell_state_tensor_2 = interpreter->output(4);

        // Store outputs and update states
        memcpy(output_sequence[step], output_tensor->data.f, 3 * sizeof(float));
        memcpy(hidden_state_1, new_hidden_state_tensor_1->data.f,
               16 * sizeof(float));
        memcpy(cell_state_1, new_cell_state_tensor_1->data.f,
               16 * sizeof(float));
        memcpy(hidden_state_2, new_hidden_state_tensor_2->data.f,
               8 * sizeof(float));
        memcpy(cell_state_2, new_cell_state_tensor_2->data.f,
               8 * sizeof(float));

        // Print step results
        MicroPrintf("Step %d Output: %.3f %.3f %.3f", step,
                    output_sequence[step][0], output_sequence[step][1],
                    output_sequence[step][2]);
    }

    // Print final sequence summary
    MicroPrintf("\n--- Sequence Summary ---");
    for (int step = 0; step < SEQUENCE_LENGTH; step++) {
        MicroPrintf("Step %d: %.3f %.3f %.3f", step, output_sequence[step][0],
                    output_sequence[step][1], output_sequence[step][2]);
    }
}
