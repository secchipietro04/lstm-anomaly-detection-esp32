#ifndef MODEL_HANDLER_H
#define MODEL_HANDLER_H

#include <stdbool.h>
#include <stddef.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
// Forward declaration
typedef struct ModelHandler ModelHandler;

// Function pointer types
namespace ModelHandlerNS{
typedef bool (*SetupFunc)(const unsigned char*, size_t);
typedef bool (*RunFunc)(void);
typedef void (*CleanupFunc)(void);
}
// Struct for model handling
struct ModelHandler {
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    uint8_t* tensor_arena;
    size_t tensor_arena_size;

    // Function pointers
    ModelHandlerNS::SetupFunc setup;
    ModelHandlerNS::RunFunc run;
    ModelHandlerNS::CleanupFunc cleanup;
};

// Initialization
void initModelHandler(ModelHandler* handler);

#endif  // MODEL_HANDLER_H
