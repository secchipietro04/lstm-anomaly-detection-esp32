idf_component_register(
    SRCS main.cc main_functions.cc sensor_interface.cc
    PRIV_REQUIRES spi_flash driver esp_timer
    INCLUDE_DIRS "")

