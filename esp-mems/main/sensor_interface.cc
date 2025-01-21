#include "sensor_interface.h"

#include "driver/i2c.h"
#include "esp_log.h"
// I2C read function
esp_err_t ism330bx_i2c_read(ism330bx_t *sensor, uint8_t reg_addr,
                            uint8_t *data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (sensor->address << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (sensor->address << 1) | I2C_MASTER_READ, true);
    i2c_master_read_byte(cmd, data, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret =
        i2c_master_cmd_begin(sensor->i2c_num, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    return ret;
}
// I2C write function
esp_err_t ism330bx_i2c_write(ism330bx_t *sensor, uint8_t reg_addr,
                             uint8_t data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (sensor->address << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret =
        i2c_master_cmd_begin(sensor->i2c_num, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    return ret;
}

// I2C burst read function
esp_err_t ism330bx_i2c_read_multi(ism330bx_t *sensor, uint8_t reg_addr,
                                  uint8_t *data, size_t length) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (sensor->address << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (sensor->address << 1) | I2C_MASTER_READ, true);
    i2c_master_read(cmd, data, length, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret =
        i2c_master_cmd_begin(sensor->i2c_num, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    return ret;
}

// Read gyroscope data with burst read
//takes 410-420 microseconds to read 
esp_err_t ism330bx_read_gyro(ism330bx_t *sensor, sensor_data_t *data) {
    uint8_t buffer[6];
    esp_err_t ret = ism330bx_i2c_read_multi(sensor, OUTX_L_G_REG, buffer, 6);
    if (ret != ESP_OK) return ret;
    
    // Combine low and high bytes
    data->gyro_x = (int16_t)((buffer[1] << 8) | buffer[0]);
    data->gyro_y = (int16_t)((buffer[3] << 8) | buffer[2]);
    data->gyro_z = (int16_t)((buffer[5] << 8) | buffer[4]);
    
    return ESP_OK;
}

// Read accelerometer data with burst read
//takes 410-420 microseconds to read 
esp_err_t ism330bx_read_accel(ism330bx_t *sensor, sensor_data_t *data) {
    uint8_t buffer[6];
    esp_err_t ret = ism330bx_i2c_read_multi(sensor, OUTX_L_A_REG, buffer, 6);
    if (ret != ESP_OK) return ret;
    
    // Combine low and high bytes
    data->acc_x = (int16_t)((buffer[1] << 8) | buffer[0]);
    data->acc_y = (int16_t)((buffer[3] << 8) | buffer[2]);
    data->acc_z = (int16_t)((buffer[5] << 8) | buffer[4]);
    
    return ESP_OK;
}

// Read timestamp data with burst read
//takes 360-370 microseconds to read 
esp_err_t ism330bx_read_timestamp(ism330bx_t *sensor, sensor_data_t *data) {
    uint8_t buffer[4];
    esp_err_t ret = ism330bx_i2c_read_multi(sensor, TIMESTAMP0_REG, buffer, 4);
    if (ret != ESP_OK) return ret;
    
    // Combine bytes into 32-bit timestamp
    data->timestamp = (uint32_t)((buffer[3] << 24) | (buffer[2] << 16) |
    (buffer[1] << 8) | buffer[0]);
    
    return ESP_OK;
}

// Initialize the sensor
esp_err_t ism330bx_init(ism330bx_t *sensor) {
    uint8_t who_am_i;
    esp_err_t ret = ism330bx_i2c_read(sensor, WHO_AM_I_REG, &who_am_i);
    if (ret != ESP_OK) {
        return ret;
    }
    ESP_LOGI("ISM330BX", "WHO_AM_I register: 0x%X", who_am_i);

    // Configure Accelerometer (CTRL1_XL)
    ret = ism330bx_i2c_write(sensor, CTRL1_XL_REG, CONFIG_ACCELERATION);
    if (ret != ESP_OK) {
        return ret;
    }

    // Configure Gyroscope (CTRL2_G)
    ret = ism330bx_i2c_write(sensor, CTRL2_G_REG, CONFIG_GYRO);
    if (ret != ESP_OK) {
        return ret;
    }

    // Configure Functions (FUNCTIONS_ENABLE)
    ret = ism330bx_i2c_write(sensor, FUNCTIONS_ENABLE_REG, FUNCTIONS_CONF);
    if (ret != ESP_OK) {
        return ret;
    }

    return ESP_OK;
}