// for models where you need to manually handle internal states

#include "rnn_model_handler.h"

#include <stdbool.h>
#include <stdlib.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static RNNModelHandler* self_handler = NULL;

// technically not needed if input shape is the same as output shape
// because tflite optimizes the memory and input addresses are the same as output addresses

// the model can do 200 hertz (per timestep)
static bool stepModel(void) {
    if (!self_handler || !self_handler->handler) {
        return false;
    }
    int i = 0;
    for (i = self_handler->inputSize; i < self_handler->inputArraySize; i++) {
        self_handler->handler->input->data.f[i] =
            self_handler->handler->output->data.f[i];
    }

    if (self_handler->handler->interpreter->Invoke() != kTfLiteOk) {
        return false;
    }
    return true;
}

// clears the input state (also the input)
static void resetModelState(void) {
    if (!self_handler) return;
    for (int i = 0; i < self_handler->inputArraySize; i++) {
        self_handler->handler->input->data.f[i] = 0.0;
    }
}

static bool setupModel(const unsigned char* model_data, size_t arena_size,
                       int inputArraySize, int outputArraySize, int inputSize,
                       int outputSize) {
    if (!self_handler) return false;
    if (!self_handler->handler->setup(model_data, arena_size)) {
        return false;
    }

    self_handler->inputArraySize = inputArraySize;
    self_handler->outputArraySize = outputArraySize;
    self_handler->inputSize = inputSize;
    self_handler->outputSize = outputSize;
    
    
    // clear the input at startup
    self_handler->reset_state();

    self_handler->input = self_handler->handler->interpreter->input(0);
    self_handler->output = self_handler->handler->interpreter->output(0);

    return true;
}

// Initialize the RNN Model Handler
void initRNNModelHandler(RNNModelHandler* RNN_handler) {
    self_handler = RNN_handler;

    // Initialize embedded ModelHandler
    RNN_handler->handler = (ModelHandler*)malloc(sizeof(ModelHandler));
    initModelHandler(RNN_handler->handler);

    RNN_handler->step = stepModel;
    RNN_handler->reset_state = resetModelState;
    RNN_handler->setup = setupModel;

    RNN_handler->cleanup = RNN_handler->handler->cleanup;
}
