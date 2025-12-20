#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "driver/gpio.h"
#include "esp_websocket_client.h"

// L298N Motor Driver Pin Suggestions
// Right Motor
#define MOTOR_RIGHT_IN1 GPIO_NUM_25
#define MOTOR_RIGHT_IN2 GPIO_NUM_26
#define MOTOR_RIGHT_ENA GPIO_NUM_14
// Left Motor
#define MOTOR_LEFT_IN3 GPIO_NUM_27
#define MOTOR_LEFT_IN4 GPIO_NUM_33
#define MOTOR_LEFT_ENB GPIO_NUM_32

#define WIFI_SSID "suaconexao"
#define WIFI_PASS "suasenha"

#define WS_SERVER_URI "ws:seusocketserver.com:8080"
#define DEVICE_ID "robot1"

static const char *TAG = "robot_ws";

/* --- Motor functions --- */
void motor_init()
{
    gpio_reset_pin(MOTOR_RIGHT_IN1);
    gpio_reset_pin(MOTOR_RIGHT_IN2);
    gpio_reset_pin(MOTOR_RIGHT_ENA);
    gpio_reset_pin(MOTOR_LEFT_IN3);
    gpio_reset_pin(MOTOR_LEFT_IN4);
    gpio_reset_pin(MOTOR_LEFT_ENB);

    gpio_set_direction(MOTOR_RIGHT_IN1, GPIO_MODE_OUTPUT);
    gpio_set_direction(MOTOR_RIGHT_IN2, GPIO_MODE_OUTPUT);
    gpio_set_direction(MOTOR_RIGHT_ENA, GPIO_MODE_OUTPUT);
    gpio_set_direction(MOTOR_LEFT_IN3, GPIO_MODE_OUTPUT);
    gpio_set_direction(MOTOR_LEFT_IN4, GPIO_MODE_OUTPUT);
    gpio_set_direction(MOTOR_LEFT_ENB, GPIO_MODE_OUTPUT);

    // Enable motors by default
    gpio_set_level(MOTOR_RIGHT_ENA, 1);
    gpio_set_level(MOTOR_LEFT_ENB, 1);

    ESP_LOGI(TAG, "Motor GPIOs initialized");
}

void stop_motion()
{
    ESP_LOGI(TAG, "Command: stop_motion");
    // Right
    gpio_set_level(MOTOR_RIGHT_IN1, 0);
    gpio_set_level(MOTOR_RIGHT_IN2, 0);
    // Left
    gpio_set_level(MOTOR_LEFT_IN3, 0);
    gpio_set_level(MOTOR_LEFT_IN4, 0);
}

void move_forward()
{
    ESP_LOGI(TAG, "Command: move_forward");
    // Right Forward
    gpio_set_level(MOTOR_RIGHT_IN1, 1);
    gpio_set_level(MOTOR_RIGHT_IN2, 0);
    // Left Forward
    gpio_set_level(MOTOR_LEFT_IN3, 0);
    gpio_set_level(MOTOR_LEFT_IN4, 1);

    vTaskDelay(pdMS_TO_TICKS(2000));
    stop_motion();
    ESP_LOGI(TAG, "Auto-stop after forward");
}

void move_backward()
{
    ESP_LOGI(TAG, "Command: move_backward");
    // Right Backward
    gpio_set_level(MOTOR_RIGHT_IN1, 0);
    gpio_set_level(MOTOR_RIGHT_IN2, 1);
    // Left Backward
    gpio_set_level(MOTOR_LEFT_IN3, 1);
    gpio_set_level(MOTOR_LEFT_IN4, 0);

    vTaskDelay(pdMS_TO_TICKS(2000));
    stop_motion();
    ESP_LOGI(TAG, "Auto-stop after backward");
}

void turn_left()
{
    ESP_LOGI(TAG, "Command: turn_left");
    // Right motor forward
    gpio_set_level(MOTOR_RIGHT_IN1, 0);
    gpio_set_level(MOTOR_RIGHT_IN2, 1);
    // Left motor backward
    gpio_set_level(MOTOR_LEFT_IN3, 0);
    gpio_set_level(MOTOR_LEFT_IN4, 1);

    vTaskDelay(pdMS_TO_TICKS(2000));
    stop_motion();
    ESP_LOGI(TAG, "Auto-stop after turn_left");
}

void turn_right()
{
    ESP_LOGI(TAG, "Command: turn_right");
    // Right motor backward
    gpio_set_level(MOTOR_RIGHT_IN1, 1);
    gpio_set_level(MOTOR_RIGHT_IN2, 0);
    // Left motor forward
    gpio_set_level(MOTOR_LEFT_IN3, 1);
    gpio_set_level(MOTOR_LEFT_IN4, 0);

    vTaskDelay(pdMS_TO_TICKS(2000));
    stop_motion();
    ESP_LOGI(TAG, "Auto-stop after turn_right");
}

/* --- WS event handler --- */
static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    esp_websocket_client_handle_t client = (esp_websocket_client_handle_t)handler_args;

    switch (event_id)
    {
    case WEBSOCKET_EVENT_CONNECTED:
        ESP_LOGI(TAG, "Websocket connected");
        esp_websocket_client_send_text(
            client,
            "register:robot:" DEVICE_ID,
            strlen("register:robot:" DEVICE_ID),
            portMAX_DELAY);
        break;

    case WEBSOCKET_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "Websocket disconnected");
        break;

    case WEBSOCKET_EVENT_DATA:
        if (data->data_len > 0)
        {
            char buf[64];
            size_t len = data->data_len < sizeof(buf) - 1 ? data->data_len : sizeof(buf) - 1;
            memcpy(buf, data->data_ptr, len);
            buf[len] = 0;

            ESP_LOGI(TAG, "Received: %s", buf);

            if (strcasecmp(buf, "FORWARD") == 0)
                move_forward();
            else if (strcasecmp(buf, "BACK") == 0 ||
                     strcasecmp(buf, "BACKWARD") == 0)
                move_backward();
            else if (strcasecmp(buf, "LEFT") == 0)
                turn_left();
            else if (strcasecmp(buf, "RIGHT") == 0)
                turn_right();
            else if (strcasecmp(buf, "STOP") == 0)
                stop_motion();
            else
                ESP_LOGI(TAG, "Unknown command: %s", buf);
        }
        break;
    }
}

/* --- Wi-Fi EVENT HANDLER --- */
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
    {
        ESP_LOGI(TAG, "Connecting to WiFi...");
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        ESP_LOGI(TAG, "Disconnected. Reconnecting...");
        esp_wifi_connect();
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

/* --- Wi-Fi Setup --- */
void wifi_init_sta(void)
{
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    // register event handlers
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                        &wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                        &wifi_event_handler, NULL, NULL);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    ESP_LOGI(TAG, "wifi_init_sta finished.");
}

/* --- app_main --- */
void app_main(void)
{
    // Init NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        nvs_flash_erase();
        nvs_flash_init();
    }

    // Init motors
    motor_init();

    // Start WiFi
    wifi_init_sta();

    // wait for WiFi to connect
    vTaskDelay(pdMS_TO_TICKS(4000));

    // Configure WebSocket
    esp_websocket_client_config_t cfg = {
        .uri = WS_SERVER_URI,
        .reconnect_timeout_ms = 5000,
    };

    esp_websocket_client_handle_t client = esp_websocket_client_init(&cfg);
    esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY,
                                  websocket_event_handler, client);

    ESP_LOGI(TAG, "Starting websocket client...");
    esp_websocket_client_start(client);

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
