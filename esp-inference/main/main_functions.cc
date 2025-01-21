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
#include "inference/model.h"
#include "inference/rnn_model_handler.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SEQUENCE_LENGTH 10
#define LSTM_UNITS 16
#define FEATURE_SIZE 3

float inputs[3];
RNNModelHandler handler;

int setupModel() {
    unsigned char* mutable_model = (unsigned char*)malloc(g_model_len);
    memcpy(mutable_model, g_model, g_model_len);


    // Initialize the handler with methods
    initRNNModelHandler(&handler);
    size_t arena_size = 1 << 16;

    if (!handler.setup( mutable_model, arena_size, 51, 51,3,3 )) {
        // Handle setup failure
        return -1;
    }
    MicroPrintf("size %d", handler.inputSize);

    return 0;
}
// The name of this function is important for Arduino compatibility.
void setup() { setupModel(); }

void loop() {
    return;
    
    int i,j;
    for (i = 0; i < 3; i++) {
        inputs[i]=-7.6;
        MicroPrintf("input: %f", inputs[i]);
    }
    for (j=0;j<1;j++){
        for (i = 0; i < 3; i++) {
            handler.input->data.f[i] = inputs[i];
            MicroPrintf("input: %f", inputs[i]);
        }
        handler.step();
        for (i = 0; i < 1; i++) {
            MicroPrintf("output: %f", handler.output->data.f[i]);
        }
        for (i = 0; i < 51; i++) {
            MicroPrintf("input after: %f", handler.input->data.f[i]);
        }
        MicroPrintf("\n");
    }

}
