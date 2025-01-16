#include "model_handler.h"
namespace RNNHandlerNS{
typedef bool (*StepFunc)(void);
typedef void (*ResetStateFunc)(void);
typedef bool (*SetupFunc)(const unsigned char*, size_t, int, int, int , int);
typedef void (*CleanupFunc)(void);
}

/*
convention for the model input and output sizes:
inputArraySize is the total size of the input array
inputSize is the size of only the input part (inputArraySize- state size)
outputArraySize is the total size of the output array
outputSize is the size of only the output part (outputArraySize-next state size)

outputSize and inputSize can be different
*/
typedef struct RNNModelHandler {
    ModelHandler* handler;
    int inputArraySize;
    int outputArraySize;
    int inputSize;
    int outputSize;
    TfLiteTensor* input;
    TfLiteTensor* output;

    RNNHandlerNS::SetupFunc setup;
    RNNHandlerNS::StepFunc step;
    RNNHandlerNS::CleanupFunc cleanup;
    RNNHandlerNS::ResetStateFunc reset_state;
} RNNModelHandler;


void initRNNModelHandler(RNNModelHandler* RNN_handler);