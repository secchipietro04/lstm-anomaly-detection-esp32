
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "main_functions.h"
#define LOOP_DELAY_MS 0
extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();

    // trigger one inference every 500ms
    vTaskDelay(pdMS_TO_TICKS(LOOP_DELAY_MS));
  }
}