idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES
        nvs_flash
        esp_netif
        esp_wifi
        esp_event
        esp_websocket_client
        esp_driver_gpio
)
