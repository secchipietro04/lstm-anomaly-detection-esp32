#include "main_functions.h"
#include "sensor_interface.h"
#include "driver/i2c.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "esp_timer.h"

/*
| esp | mki   |
|-----|-------|
| 9   | sda   |
| 10  | scl   |
| gnd | sdo   |
| gnd | gnd   |
| 3.3 | cs    |
| 3.3 | vdd   |
| 3.3 | vddio |
*/

#define TAG "ISM330BX"


// Sensor instance
static ism330bx_t sensor;

// Sensor data
static sensor_data_t data;

void setup() {
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
        return;
    }

    ret = i2c_driver_install(I2C_MASTER_NUM, i2c_config.mode, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install I2C driver");
        return;
    }

    // Create ISM330BX sensor instance
    sensor.i2c_num = I2C_MASTER_NUM;
    sensor.address = ISM330BX_I2C_ADDR;

    // Initialize the sensor
    ret = ism330bx_init(&sensor);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize ISM330BX sensor");
        return;
    }

    ESP_LOGI(TAG, "ISM330BX sensor initialized successfully");
}

void loop() {
    esp_err_t ret;

    int64_t start_time = esp_timer_get_time();
    // Read gyroscope data
    //takes 1180-1190 microseconds to read all three (9 values total)
    
    ret = ism330bx_read_gyro(&sensor, &data);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read gyroscope data");
        return;
    }

    int64_t end_time = esp_timer_get_time();
    printf("Start time  Gyro: %lld microseconds\n", start_time);
    printf("End time  Gyro: %lld microseconds\n", end_time);

    // Calculate and print the elapsed time
    printf("Elapsed time Gyro: %lld microseconds\n", end_time - start_time);
    start_time = esp_timer_get_time();
    // Read accelerometer data
    ret = ism330bx_read_accel(&sensor, &data);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read accelerometer data");
        return;
    }
    end_time = esp_timer_get_time();
    printf("Start time Accell: %lld microseconds\n", start_time);
    printf("End time Accell: %lld microseconds\n", end_time);
    printf("Elapsed time Accell: %lld microseconds\n", end_time - start_time);

    
    // Read timestamp
    start_time = esp_timer_get_time();
    ret = ism330bx_read_timestamp(&sensor, &data);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read timestamp");
        return;
    }
     end_time = esp_timer_get_time();
    printf("Start time timestamp: %lld microseconds\n", start_time);
    printf("End time timestamp: %lld microseconds\n", end_time);

    // Calculate and print the elapsed time
    printf("Elapsed time timestamp: %lld microseconds\n", end_time - start_time);

    // Print the data
    ESP_LOGI(TAG, "Gyro: X=%d, Y=%d, Z=%d | Accel: X=%d, Y=%d, Z=%d | Timestamp: %lu",
             data.gyro_x, data.gyro_y, data.gyro_z,
             data.acc_x, data.acc_y, data.acc_z,
             data.timestamp);

    // Delay to prevent overwhelming the log
    vTaskDelay(pdMS_TO_TICKS(100));
}