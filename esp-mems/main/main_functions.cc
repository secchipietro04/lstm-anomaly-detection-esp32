#include "main_functions.h"

#include "driver/i2c.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"

/*
| esp | mki   |
|-----|-------|
| 41  | sda   |
| 42  | scl   |
| gnd | sdo   |
| gnd | gnd   |
| 3.3 | cs    |
| 3.3 | vdd   |
| 3.3 | vddio |
*/

#define TCP_PORT 8080
#define TCP_BUFFER_SIZE 128
#define TAG "ISM330BX"

// I2C configuration
#define I2C_MASTER_NUM I2C_NUM_0
#define I2C_MASTER_SCL_IO 42
#define I2C_MASTER_SDA_IO 41
#define I2C_MASTER_FREQ_HZ 100000
#define I2C_MASTER_TX_BUF_DISABLE 0
#define I2C_MASTER_RX_BUF_DISABLE 0
#define ISM330BX_I2C_ADDR 0x6A  // Change to 0x6B if SDO is high

// ISM330BX Register Addresses
#define WHO_AM_I_REG 0x0F
#define CTRL1_XL_REG 0x10
#define CTRL2_G_REG 0x11

#define OUTX_L_G_REG 0x22  // Register address for OUTX_L_G
#define OUTX_H_G_REG 0x23  // Register address for OUTX_H_G
#define OUTY_L_G_REG 0x24  // Register address for OUTX_L_G
#define OUTY_H_G_REG 0x25  // Register address for OUTX_H_G
#define OUTZ_L_G_REG 0x26  // Register address for OUTX_L_G
#define OUTZ_H_G_REG 0x27

#define OUTZ_L_A_REG 0x28  // Register address for OUTX_L_G
#define OUTZ_H_A_REG 0x29  // Register address for OUTX_H_G
#define OUTY_L_A_REG 0x2a
#define OUTY_H_A_REG 0x2b
#define OUTX_L_A_REG 0x2c
#define OUTX_H_A_REG 0x2d

#define UI_OUTZ_L_A_DualC 0x34  // Register address for OUTX_L_G
#define UI_OUTZ_H_A_DualC 0x35  // Register address for OUTX_H_G
#define UI_OUTY_L_A_DualC 0x36
#define UI_OUTY_H_A_DualC 0x37
#define UI_OUTX_L_A_DualC 0x38
#define UI_OUTX_H_A_DualC 0x39

#define LOOP_DELAY_MS 0

#define UART_NUM UART_NUM_1
#define BUF_SIZE 1024
void uart_init() {
    uart_config_t uart_config = {.baud_rate = 115200,
                                 .data_bits = UART_DATA_8_BITS,
                                 .parity = UART_PARITY_DISABLE,
                                 .stop_bits = UART_STOP_BITS_1,
                                 .flow_ctrl = UART_HW_FLOWCTRL_DISABLE};
    uart_param_config(UART_NUM, &uart_config);
    uart_set_pin(UART_NUM, 17, 16, UART_PIN_NO_CHANGE,
                 UART_PIN_NO_CHANGE);  // TX=17, RX=16
    uart_driver_install(UART_NUM, BUF_SIZE, BUF_SIZE, 0, NULL, 0);
}

// Helper to write to I2C
esp_err_t i2c_master_write_byte(i2c_port_t i2c_num, uint8_t reg_addr,
                                uint8_t data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (ISM330BX_I2C_ADDR << 1) | I2C_MASTER_WRITE,
                          true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_write_byte(cmd, data, true);
    i2c_master_stop(cmd);
    esp_err_t ret =
        i2c_master_cmd_begin(i2c_num, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    return ret;
}

// Helper to read from I2C
esp_err_t i2c_master_read_byte(i2c_port_t i2c_num, uint8_t reg_addr,
                               uint8_t *data) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (ISM330BX_I2C_ADDR << 1) | I2C_MASTER_WRITE,
                          true);
    i2c_master_write_byte(cmd, reg_addr, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (ISM330BX_I2C_ADDR << 1) | I2C_MASTER_READ,
                          true);
    i2c_master_read_byte(cmd, data, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret =
        i2c_master_cmd_begin(i2c_num, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    return ret;
}
void ism330bx_init() {
    uint8_t who_am_i;
    ESP_ERROR_CHECK(
        i2c_master_read_byte(I2C_MASTER_NUM, WHO_AM_I_REG, &who_am_i));
    ESP_LOGI(TAG, "WHO_AM_I register: 0x%X", who_am_i);

    // Configure Accelerometer (CTRL1_XL)
    ESP_ERROR_CHECK(i2c_master_write_byte(I2C_MASTER_NUM, CTRL1_XL_REG, 0x8));
    // Configure Gyroscope (CTRL2_G)
    ESP_ERROR_CHECK(i2c_master_write_byte(I2C_MASTER_NUM, CTRL2_G_REG, 0x8));
}

void i2c_scan() {
    printf("Scanning I2C bus...\n");
    for (uint8_t addr = 127; addr > 0; addr--) {
        ESP_LOGI(TAG, "Scanning address: 0x%02X", addr);
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_stop(cmd);
        esp_err_t ret =
            i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, 500 / portTICK_PERIOD_MS);
        i2c_cmd_link_delete(cmd);

        if (ret == ESP_OK) {
            printf("Found device at address 0x%02X\n", addr);
        }
    }
    printf("Scan complete.\n");
}

int16_t read_gyro(uint8_t reg_h, uint8_t reg_l) {
    uint8_t outx_l, outx_h;
    esp_err_t ret;

    // Read low byte of gyroscope X-axis
    ret = i2c_master_read_byte(I2C_MASTER_NUM, reg_l, &outx_l);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read OUTX_L_G");
        return 0;
    }

    // Read high byte of gyroscope X-axis
    ret = i2c_master_read_byte(I2C_MASTER_NUM, reg_h, &outx_h);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read OUTX_H_G");
        return 0;
    }

    // Combine high and low bytes into a single 16-bit value
    int16_t gyro_x = (int16_t)((outx_h << 8) | outx_l);
    return gyro_x;
}

int16_t read_accel_x() {
    uint8_t outx_l, outx_h;
    esp_err_t ret;

    // Read low byte of gyroscope X-axis
    ret = i2c_master_read_byte(I2C_MASTER_NUM, OUTX_L_A_REG, &outx_l);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read OUTX_L_G");
        return 0;
    }

    // Read high byte of gyroscope X-axis
    ret = i2c_master_read_byte(I2C_MASTER_NUM, OUTX_H_A_REG, &outx_h);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read OUTX_H_G");
        return 0;
    }

    // Combine high and low bytes into a single 16-bit value
    int16_t gyro_x = (int16_t)((outx_h << 8) | outx_l);
    ESP_LOGI(TAG, "%d %d", outx_h, outx_l);
    return gyro_x;
}

uint8_t accel_data[6];
uint8_t gyro_data[6];
void setup() {
    i2c_config_t i2c_master_config = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {.clk_speed = I2C_MASTER_FREQ_HZ},
        .clk_flags = 0,

    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &i2c_master_config));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, I2C_MODE_MASTER,
                                       I2C_MASTER_RX_BUF_DISABLE,
                                       I2C_MASTER_TX_BUF_DISABLE, 0));
    // i2c_scan();
    ism330bx_init();
    uart_init();
    // i2c_scanner();
}

int16_t gyro_max = 0;
int16_t gyro_min = 0;
int16_t acc_max = 0;
int16_t acc_min = 0;
int16_t lin_acc_max = 0;
int16_t lin_acc_min = 0;

void loop() {
    int16_t gyro_x = read_gyro(OUTX_H_G_REG, OUTX_L_G_REG);
    int16_t gyro_y = read_gyro(OUTY_H_G_REG, OUTY_L_G_REG);
    int16_t gyro_z = read_gyro(OUTZ_H_G_REG, OUTZ_L_G_REG);

    int16_t acc_x = read_gyro(OUTX_H_A_REG, OUTX_L_A_REG);
    int16_t acc_y = read_gyro(OUTY_H_A_REG, OUTY_L_A_REG);
    int16_t acc_z = read_gyro(OUTZ_H_A_REG, OUTZ_L_A_REG);

    int16_t lin_acc_x = read_gyro(UI_OUTX_H_A_DualC, UI_OUTX_L_A_DualC);
    int16_t lin_acc_y = read_gyro(UI_OUTY_H_A_DualC, UI_OUTY_L_A_DualC);
    int16_t lin_acc_z = read_gyro(UI_OUTZ_H_A_DualC, UI_OUTZ_L_A_DualC);

    if (gyro_x > gyro_max) {
        gyro_max = gyro_x;
    }
    if (gyro_x < gyro_min) {
        gyro_min = gyro_x;
    }
    if (acc_x > acc_max) {
        acc_max = acc_x;
    }
    if (acc_x < acc_min) {
        acc_min = acc_x;
    }
    if (lin_acc_x > lin_acc_max) {
        lin_acc_max = lin_acc_x;
    }
    if (lin_acc_x < lin_acc_min) {
        lin_acc_min = lin_acc_x;
    }

    ESP_LOGI(TAG, "-1,Gyro Max: %d Min: %d", gyro_max, gyro_min);
    ESP_LOGI(TAG, "-1,Acc Max: %d Min: %d", acc_max, acc_min);
    ESP_LOGI(TAG, "-1,Lin Acc Max: %d Min: %d", lin_acc_max, lin_acc_min);

    char buffer[128];
    int len =
        snprintf(buffer, sizeof(buffer), "%d,%d,%d\n", gyro_x, gyro_y, gyro_z);
    uart_write_bytes(UART_NUM, buffer, len);
    ESP_LOGI(TAG, "0,%d,%d,%d", gyro_x, gyro_y, gyro_z);
    ESP_LOGI(TAG, "1,%d,%d,%d", acc_x, acc_y, acc_z);
    ESP_LOGI(TAG, "2,%d,%d,%d", lin_acc_x, lin_acc_y, lin_acc_z);
    vTaskDelay(pdMS_TO_TICKS(LOOP_DELAY_MS));
}
