#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "mqtt_client.h"
#include "cJSON.h"
#include "esp_crt_bundle.h"

/* Configuration constants */
#define WIFI_SSID "IZZI-1038"
#define WIFI_PASSWORD "704FB8271038"
#define ABLY_API_KEY "ja_fwQ.K6CTEw:F-aWFMdJXPCv9MvxhYztCGna3XdRJZVgA0qm9pMfDOQ"
#define ABLY_CLIENT_ID_PREFIX "esp32_telemetry_"
#define ABLY_CHANNEL "EcoTele"
#define PUBLISH_INTERVAL 5000  /* 5 seconds in milliseconds */

/* Ably MQTT configuration - SSL version (REQUIRED for API key auth) */
#define MQTT_BROKER_HOST "mqtt.ably.io"
#define MQTT_BROKER_PORT 8883  // SSL port
#define MQTT_USERNAME ABLY_API_KEY
#define MQTT_PASSWORD ""

static const char *TAG = "TELEMETRY_SIM";

/* Event group for WiFi and MQTT status */
static EventGroupHandle_t wifi_event_group;
static EventGroupHandle_t mqtt_event_group;
const int WIFI_CONNECTED_BIT = BIT0;
const int MQTT_CONNECTED_BIT = BIT0;

static esp_mqtt_client_handle_t mqtt_client = NULL;

/* Telemetry simulation state */
typedef struct {
    float simulation_time;
    float cumulative_energy;
    float cumulative_distance;
    float vehicle_heading;
    float prev_speed;
    int message_count;
    int64_t start_time_us;
} telemetry_state_t;

static telemetry_state_t telemetry_state = {0};

/* Random number generator helpers */
static float random_gaussian(float mean, float stddev) {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare * stddev + mean;
    }
    
    has_spare = 1;
    static float u, v, mag;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        mag = u * u + v * v;
    } while (mag >= 1.0f || mag == 0.0f);
    
    mag = sqrtf(-2.0f * logf(mag) / mag);
    spare = v * mag;
    return (u * mag) * stddev + mean;
}

static float clamp_f(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/* Get current timestamp in ISO format */
static void get_iso_timestamp(char *buffer, size_t buffer_size) {
    struct timeval tv;
    struct tm timeinfo;
    
    gettimeofday(&tv, NULL);
    gmtime_r(&tv.tv_sec, &timeinfo);
    
    snprintf(buffer, buffer_size, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
             timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
             timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec,
             tv.tv_usec / 1000);
}

/* Generate unique client ID using MAC address */
static void generate_client_id(char *client_id, size_t max_len) {
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    snprintf(client_id, max_len, "%s%02x%02x%02x%02x%02x%02x", 
             ABLY_CLIENT_ID_PREFIX, 
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

/* Generate telemetry data (C equivalent of Python function) */
static cJSON* generate_telemetry_data(telemetry_state_t *state) {
    char timestamp[32];
    get_iso_timestamp(timestamp, sizeof(timestamp));
    
    /* Generate realistic speed (0-25 m/s with variations) */
    float base_speed = 15.0f + 5.0f * sinf(state->simulation_time * 0.1f);
    float speed_variation = random_gaussian(0, 1.5f);
    float speed = clamp_f(base_speed + speed_variation, 0.0f, 25.0f);
    
    /* Generate electrical system data */
    float voltage = clamp_f(48.0f + random_gaussian(0, 1.5f), 40.0f, 55.0f);
    float current = clamp_f(8.0f + speed * 0.2f + random_gaussian(0, 1.0f), 0.0f, 15.0f);
    float power = voltage * current;
    
    /* Accumulate energy and distance */
    float energy_delta = power * (PUBLISH_INTERVAL / 1000.0f);
    float distance_delta = speed * (PUBLISH_INTERVAL / 1000.0f);
    
    state->cumulative_energy += energy_delta;
    state->cumulative_distance += distance_delta;
    
    /* Generate GPS coordinates (simulated route) */
    float base_lat = 40.7128f;
    float base_lon = -74.0060f;
    float lat_offset = 0.001f * sinf(state->simulation_time * 0.05f);
    float lon_offset = 0.001f * cosf(state->simulation_time * 0.05f);
    
    float latitude = base_lat + lat_offset + random_gaussian(0, 0.0001f);
    float longitude = base_lon + lon_offset + random_gaussian(0, 0.0001f);
    
    /* Generate Gyroscope data (angular velocity in deg/s) */
    float turning_rate = 2.0f * sinf(state->simulation_time * 0.08f);
    float gyro_x = random_gaussian(0, 0.5f);
    float gyro_y = random_gaussian(0, 0.3f);
    float gyro_z = turning_rate + random_gaussian(0, 0.8f);
    
    /* Update vehicle heading */
    state->vehicle_heading += gyro_z * (PUBLISH_INTERVAL / 1000.0f);
    
    /* Generate Accelerometer data (m/s2) */
    float speed_acceleration = (speed - state->prev_speed) / (PUBLISH_INTERVAL / 1000.0f);
    state->prev_speed = speed;
    
    float accel_x = speed_acceleration + random_gaussian(0, 0.2f);
    float accel_y = turning_rate * speed * 0.1f + random_gaussian(0, 0.1f);
    float accel_z = 9.81f + random_gaussian(0, 0.05f);
    
    /* Add vibrations correlated with speed */
    float vibration_factor = speed * 0.02f;
    accel_x += random_gaussian(0, vibration_factor);
    accel_y += random_gaussian(0, vibration_factor);
    accel_z += random_gaussian(0, vibration_factor);
    
    /* Calculate total acceleration */
    float total_acceleration = sqrtf(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z);
    
    /* Get uptime in seconds */
    int64_t current_time_us = esp_timer_get_time();
    float uptime_seconds = (current_time_us - state->start_time_us) / 1000000.0f;
    
    state->simulation_time += 1.0f;
    state->message_count++;
    
    /* Create JSON object */
    cJSON *json = cJSON_CreateObject();
    
    cJSON_AddStringToObject(json, "timestamp", timestamp);
    cJSON_AddNumberToObject(json, "speed_ms", roundf(speed * 100) / 100);
    cJSON_AddNumberToObject(json, "voltage_v", roundf(voltage * 100) / 100);
    cJSON_AddNumberToObject(json, "current_a", roundf(current * 100) / 100);
    cJSON_AddNumberToObject(json, "power_w", roundf(power * 100) / 100);
    cJSON_AddNumberToObject(json, "energy_j", roundf(state->cumulative_energy * 100) / 100);
    cJSON_AddNumberToObject(json, "distance_m", roundf(state->cumulative_distance * 100) / 100);
    cJSON_AddNumberToObject(json, "latitude", roundf(latitude * 1000000) / 1000000);
    cJSON_AddNumberToObject(json, "longitude", roundf(longitude * 1000000) / 1000000);
    cJSON_AddNumberToObject(json, "gyro_x", roundf(gyro_x * 1000) / 1000);
    cJSON_AddNumberToObject(json, "gyro_y", roundf(gyro_y * 1000) / 1000);
    cJSON_AddNumberToObject(json, "gyro_z", roundf(gyro_z * 1000) / 1000);
    cJSON_AddNumberToObject(json, "accel_x", roundf(accel_x * 1000) / 1000);
    cJSON_AddNumberToObject(json, "accel_y", roundf(accel_y * 1000) / 1000);
    cJSON_AddNumberToObject(json, "accel_z", roundf(accel_z * 1000) / 1000);
    cJSON_AddNumberToObject(json, "vehicle_heading", roundf(fmodf(state->vehicle_heading, 360.0f) * 100) / 100);
    cJSON_AddNumberToObject(json, "total_acceleration", roundf(total_acceleration * 1000) / 1000);
    cJSON_AddNumberToObject(json, "message_id", state->message_count);
    cJSON_AddNumberToObject(json, "uptime_seconds", roundf(uptime_seconds * 100) / 100);
    
    return json;
}

/* MQTT event handler */
static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    esp_mqtt_client_handle_t client = event->client;
    
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
            xEventGroupSetBits(mqtt_event_group, MQTT_CONNECTED_BIT);
            break;
            
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
            xEventGroupClearBits(mqtt_event_group, MQTT_CONNECTED_BIT);
            break;
            
        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED, msg_id=%d", event->msg_id);
            break;
            
        case MQTT_EVENT_UNSUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
            break;
            
        case MQTT_EVENT_PUBLISHED:
            ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
            break;
            
        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "MQTT_EVENT_DATA");
            ESP_LOGI(TAG, "TOPIC=%.*s", event->topic_len, event->topic);
            ESP_LOGI(TAG, "DATA=%.*s", event->data_len, event->data);
            break;
            
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT_EVENT_ERROR");
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGE(TAG, "Last errno string (%s)", strerror(event->error_handle->esp_transport_sock_errno));
            }
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                ESP_LOGE(TAG, "Connection refused error: 0x%x", event->error_handle->connect_return_code);
            }
            break;
            
        default:
            ESP_LOGI(TAG, "Other event id:%d", event->event_id);
            break;
    }
}

/* Initialize MQTT client with SSL/TLS */
static void mqtt_app_start(void) {
    static char client_id[64];
    generate_client_id(client_id, sizeof(client_id));
    
    ESP_LOGI(TAG, "Using client ID: %s", client_id);
    
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .hostname = MQTT_BROKER_HOST,
                .port = MQTT_BROKER_PORT,
                .transport = MQTT_TRANSPORT_OVER_SSL,  // SSL transport required
            },
            .verification = {
                .skip_cert_common_name_check = true,   // Skip hostname verification
                .crt_bundle_attach = esp_crt_bundle_attach,  // Use certificate bundle
            },
        },
        .credentials = {
            .username = MQTT_USERNAME,
            .authentication = {
                .password = MQTT_PASSWORD,
            },
            .client_id = client_id,
        },
        .session = {
            .keepalive = 60,
            .disable_clean_session = false,
        },
        .network = {
            .timeout_ms = 10000,
            .refresh_connection_after_ms = 20000,
            .disable_auto_reconnect = false,
        },
    };
    
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (mqtt_client == NULL) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return;
    }
    
    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    
    esp_err_t start_result = esp_mqtt_client_start(mqtt_client);
    if (start_result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start MQTT client: %s", esp_err_to_name(start_result));
    }
}

/* Send telemetry data via MQTT */
static void send_telemetry_mqtt(cJSON *json_data) {
    if (mqtt_client == NULL) {
        ESP_LOGE(TAG, "MQTT client not initialized");
        return;
    }
    
    /* Wait for MQTT connection with timeout */
    EventBits_t bits = xEventGroupWaitBits(mqtt_event_group, MQTT_CONNECTED_BIT, 
                                          pdFALSE, pdFALSE, pdMS_TO_TICKS(2000));
    
    if (!(bits & MQTT_CONNECTED_BIT)) {
        ESP_LOGW(TAG, "MQTT not connected, skipping publish");
        return;
    }
    
    char *json_string = cJSON_Print(json_data);
    if (json_string == NULL) {
        ESP_LOGE(TAG, "Failed to create JSON string");
        return;
    }
    
    /* Ably MQTT topic format: just the channel name */
    char topic[128];
    snprintf(topic, sizeof(topic), "%s", ABLY_CHANNEL);
    
    int msg_id = esp_mqtt_client_publish(mqtt_client, topic, json_string, 0, 0, 0);
    
    if (msg_id >= 0) {
        ESP_LOGI(TAG, "Telemetry published successfully, msg_id=%d", msg_id);
        ESP_LOGI(TAG, "Published to topic: %s", topic);
    } else {
        ESP_LOGE(TAG, "Failed to publish telemetry data");
    }
    
    free(json_string);
}

/* WiFi event handler */
static void event_handler(void* arg, esp_event_base_t event_base,
                         int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGI(TAG, "retry to connect to the AP");
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* Initialize WiFi */
static void wifi_init_sta(void) {
    wifi_event_group = xEventGroupCreate();
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));
    
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    ESP_LOGI(TAG, "wifi_init_sta finished.");
}

/* Main telemetry task */
static void telemetry_task(void *pvParameters) {
    /* Wait for WiFi connection */
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
    ESP_LOGI(TAG, "WiFi connected, starting MQTT...");
    
    /* Initialize MQTT */
    mqtt_event_group = xEventGroupCreate();
    mqtt_app_start();
    
    /* Wait for MQTT connection with timeout */
    ESP_LOGI(TAG, "Waiting for MQTT connection...");
    EventBits_t bits = xEventGroupWaitBits(mqtt_event_group, MQTT_CONNECTED_BIT, 
                                          pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));
    
    if (bits & MQTT_CONNECTED_BIT) {
        ESP_LOGI(TAG, "MQTT connected!");
    } else {
        ESP_LOGE(TAG, "MQTT connection timeout! Continuing anyway...");
    }
    
    /* Initialize random seed and telemetry state */
    srand(time(NULL));
    telemetry_state.start_time_us = esp_timer_get_time();
    
    ESP_LOGI(TAG, "Starting telemetry simulation...");
    
    while (1) {
        /* Generate telemetry data */
        cJSON *telemetry_json = generate_telemetry_data(&telemetry_state);
        
        /* Send via MQTT */
        send_telemetry_mqtt(telemetry_json);
        
        /* Log telemetry data locally */
        char *json_string = cJSON_Print(telemetry_json);
        ESP_LOGI(TAG, "Generated telemetry: %s", json_string);
        
        /* Cleanup */
        free(json_string);
        cJSON_Delete(telemetry_json);
        
        /* Wait for next iteration */
        vTaskDelay(pdMS_TO_TICKS(PUBLISH_INTERVAL));
    }
}

void app_main(void) {
    /* Initialize NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    ESP_LOGI(TAG, "Initializing ESP32 Telemetry Simulator with MQTT over SSL");
    wifi_init_sta();
    
    /* Create telemetry task */
    xTaskCreate(telemetry_task, "telemetry_task", 8192, NULL, 5, NULL);
}
