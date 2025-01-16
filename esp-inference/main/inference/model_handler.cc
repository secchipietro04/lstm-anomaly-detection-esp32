#include "model_handler.h"
#include <stdlib.h>

// Internal static handler for function callbacks
static ModelHandler* self_handler = NULL;

// Setup function
static bool setupModel(const unsigned char* model_data, size_t arena_size) {
    if (!self_handler) return false;

    self_handler->model = tflite::GetModel(model_data);
    if (self_handler->model->version() != TFLITE_SCHEMA_VERSION) {
        return false;  // Model version mismatch
    }

    // Allocate tensor arena dynamically
    self_handler->tensor_arena = (uint8_t*)malloc(arena_size);
    if (self_handler->tensor_arena == NULL) {
        return false;  // Memory allocation failed
    }
    self_handler->tensor_arena_size = arena_size;

    // Register necessary operators (adjust as needed)
    static tflite::MicroMutableOpResolver<50> resolver;
    if (resolver.AddRelu() != kTfLiteOk) return false;
    if (resolver.AddSoftmax() != kTfLiteOk) return false;
    if (resolver.AddQuantize() != kTfLiteOk) return false;
    if (resolver.AddDequantize() != kTfLiteOk) return false;
    if (resolver.AddAdd() != kTfLiteOk) return false;
    if (resolver.AddConcatenation() != kTfLiteOk) return false;
    if (resolver.AddFullyConnected() != kTfLiteOk) return false;
    if (resolver.AddLogistic() != kTfLiteOk) return false;
    if (resolver.AddMul() != kTfLiteOk) return false;
    if (resolver.AddSlice() != kTfLiteOk) return false;
    if (resolver.AddSplit() != kTfLiteOk) return false;
    if (resolver.AddTanh() != kTfLiteOk) return false;

    // Create interpreter
    static tflite::MicroInterpreter interpreter(
        self_handler->model, resolver, self_handler->tensor_arena,
        self_handler->tensor_arena_size);
    self_handler->interpreter = &interpreter;

    if (self_handler->interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }

    self_handler->input = self_handler->interpreter->input(0);
    self_handler->output = self_handler->interpreter->output(0);

    return true;
}


// Run function
static bool runModel(void) {
    if (!self_handler || self_handler->interpreter->Invoke() != kTfLiteOk) {
        return false;
    }
    return true;
}

// Cleanup function
static void cleanupModel(void) {
    if (self_handler && self_handler->tensor_arena) {
        free(self_handler->tensor_arena);
        self_handler->tensor_arena = NULL;
    }
}

// Initialize the handler
void initModelHandler(ModelHandler* handler) {
    self_handler = handler;
    handler->setup = setupModel;
    handler->run = runModel;
    handler->cleanup = cleanupModel;
}
