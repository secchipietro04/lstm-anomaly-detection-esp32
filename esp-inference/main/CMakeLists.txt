
#
# Main component of TF Micro project 'micro_speech'.
#



idf_component_register(
    SRCS  "main.cc"
         "main_functions.cc"
         "constants.cc"
         "output_handler.cc"
         "inference/model_handler.cc"
         "inference/rnn_model_handler.cc"
         "inference/model.cc"
         "sensor/sensor_interface.cc"
    PRIV_REQUIRES spi_flash esp_timer driver
    INCLUDE_DIRS "."
)

