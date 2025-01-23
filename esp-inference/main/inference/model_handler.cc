#include "model_handler.h"

#include <stdlib.h>
#include "esp_log.h"
// Internal static handler for function callbacks
static ModelHandler* self_handler = NULL;
static tflite::MicroMutableOpResolver<50> resolver;
static bool already_initialized = false;
#define TAG "MODEL_HANDLER"
// Setup function
static bool setupModel(const unsigned char* model_data, size_t arena_size) {
    if (!self_handler) {
        ESP_LOGE(TAG, "self_handler is null.");
        return false;
    }

    // Log model_data pointer to check if it's valid
    ESP_LOGI(TAG, "Model data address: %p", model_data);

    // Get the model and validate
    self_handler->model = tflite::GetModel(model_data);
    if (!self_handler->model) {
        ESP_LOGE(TAG, "Failed to get the model from model_data.");
        return false;
    }

    // Validate the model schema version
    if (self_handler->model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: Expected %d, Got %lu", 
                 TFLITE_SCHEMA_VERSION, self_handler->model->version());
        return false;
    }

    // Initialize resolver if not already initialized
    if (!already_initialized) {
        ESP_LOGI(TAG, "Initializing operator resolver...");
        
        if (resolver.AddRelu() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add ReLU operator.");
            return false;
        }
        if (resolver.AddSoftmax() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Softmax operator.");
            return false;
        }
        if (resolver.AddQuantize() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Quantize operator.");
            return false;
        }
        if (resolver.AddDequantize() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Dequantize operator.");
            return false;
        }
        if (resolver.AddAdd() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Add operator.");
            return false;
        }
        if (resolver.AddConcatenation() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Concatenation operator.");
            return false;
        }
        if (resolver.AddFullyConnected() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add FullyConnected operator.");
            return false;
        }
        if (resolver.AddLogistic() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Logistic operator.");
            return false;
        }
        if (resolver.AddMul() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Mul operator.");
            return false;
        }
        if (resolver.AddSlice() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Slice operator.");
            return false;
        }
        if (resolver.AddSplit() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Split operator.");
            return false;
        }
        if (resolver.AddTanh() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to add Tanh operator.");
            return false;
        }

        already_initialized = true;
    }

    // Allocate tensor arena dynamically
    ESP_LOGI(TAG, "Allocating tensor arena of size: %zu", arena_size);
    self_handler->tensor_arena = (uint8_t*)malloc(arena_size);
    if (self_handler->tensor_arena == NULL) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena. Requested size: %zu", arena_size);
        return false;
    }
    self_handler->tensor_arena_size = arena_size;

    // Create interpreter
    ESP_LOGI(TAG, "Creating interpreter...");
    tflite::MicroInterpreter* interpreter = 
        new tflite::MicroInterpreter(self_handler->model, resolver, 
                                     self_handler->tensor_arena, 
                                     self_handler->tensor_arena_size);

    if (!interpreter) {
        ESP_LOGE(TAG, "Failed to create MicroInterpreter.");
        free(self_handler->tensor_arena);
        self_handler->tensor_arena = NULL;
        return false;
    }
    self_handler->interpreter = interpreter;

    // Allocate tensors
    ESP_LOGI(TAG, "Allocating tensors...");
    if (self_handler->interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Tensor allocation failed.");
        delete self_handler->interpreter;
        self_handler->interpreter = NULL;
        free(self_handler->tensor_arena);
        self_handler->tensor_arena = NULL;
        return false;
    }

    // Get input and output tensors
    ESP_LOGI(TAG, "Fetching input and output tensors...");
    self_handler->input = self_handler->interpreter->input(0);
    if (!self_handler->input) {
        ESP_LOGE(TAG, "Failed to get input tensor.");
        delete self_handler->interpreter;
        self_handler->interpreter = NULL;
        free(self_handler->tensor_arena);
        self_handler->tensor_arena = NULL;
        return false;
    }
    self_handler->output = self_handler->interpreter->output(0);
    if (!self_handler->output) {
        ESP_LOGE(TAG, "Failed to get output tensor.");
        delete self_handler->interpreter;
        self_handler->interpreter = NULL;
        free(self_handler->tensor_arena);
        self_handler->tensor_arena = NULL;
        return false;
    }

    ESP_LOGI(TAG, "Model setup successful.");
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
