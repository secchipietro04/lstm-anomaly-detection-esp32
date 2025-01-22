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
#include "esp_log.h"
#include "esp_timer.h"
#include "inference/model.h"
#include "inference/rnn_model_handler.h"
#include "output_handler.h"
#include "sensor/sensor_interface.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SEQUENCE_LENGTH 10
#define LSTM_UNITS 16
#define FEATURE_SIZE 3

#define TAG "ISM330BX"

float inputs[3];
RNNModelHandler handler;

int setupModel() {
    unsigned char* mutable_model = (unsigned char*)malloc(g_model_len);
    memcpy(mutable_model, g_model, g_model_len);

    // Initialize the handler with methods
    initRNNModelHandler(&handler);
    size_t arena_size = 1 << 16;

    if (!handler.setup(mutable_model, arena_size, 51, 51, 3, 3)) {
        // Handle setup failure
        return -1;
    }
    MicroPrintf("size %d", handler.inputSize);

    return 0;
}

// Sensor instance
static ism330bx_t sensor;

// Sensor data
static sensor_data_t data;

esp_err_t ret;

int setupSensor() {
    // Initialize I2C driver
    i2c_config_t i2c_config = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {.clk_speed = I2C_MASTER_FREQ_HZ},
        .clk_flags = 0,
    };

    esp_err_t ret = i2c_param_config(I2C_MASTER_NUM, &i2c_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure I2C");
        return -1;
    }

    ret = i2c_driver_install(I2C_MASTER_NUM, i2c_config.mode, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install I2C driver");
        return -1;
    }

    // Create ISM330BX sensor instance
    sensor.i2c_num = I2C_MASTER_NUM;
    sensor.address = ISM330BX_I2C_ADDR;

    // Initialize the sensor
    ret = ism330bx_init(&sensor);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize ISM330BX sensor");
        return -1;
    }

    ESP_LOGI(TAG, "ISM330BX sensor initialized successfully");
    return 0;
}

void setup() {
    setupModel();
    setupSensor();
}

void loop() {
    esp_err_t ret;



    ret = ism330bx_read_gyro(&sensor, &data);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read gyroscope data");
        return;
    }
    ESP_LOGI(TAG, "Gyro: X=%d, Y=%d, Z=%d",
        data.gyro_x, data.gyro_y, data.gyro_z
        );
}
