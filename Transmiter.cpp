
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <random>
#include <memory>
#include <sstream>
#include <iomanip>
#include <sys/time.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "mqtt_client.h"
#include "cJSON.h"
#include "esp_crt_bundle.h"

namespace TelemetryConfig {
    const char* WIFI_SSID = "IZZI-1038";
    const char* WIFI_PASSWORD = "704FB8271038";
    const char* ABLY_API_KEY = "ja_fwQ.K6CTEw:F-aWFMdJXPCv9MvxhYztCGna3XdRJZVgA0qm9pMfDOQ";
    const char* ABLY_CLIENT_ID_PREFIX = "esp32_telemetry_";
    const char* ABLY_CHANNEL = "EcoTele";
    const uint32_t PUBLISH_INTERVAL = 5000; // 5 seconds in milliseconds
    
    // Ably MQTT configuration - SSL version (REQUIRED for API key auth)
    const char* MQTT_BROKER_HOST = "mqtt.ably.io";
    const int MQTT_BROKER_PORT = 8883; // SSL port
    const char* MQTT_USERNAME = ABLY_API_KEY;
    const char* MQTT_PASSWORD = "";
}

class TelemetryData {
public:
    struct SensorData {
        float speed_ms;
        float voltage_v;
        float current_a;
        float power_w;
        float energy_j;
        float distance_m;
        double latitude;
        double longitude;
        float gyro_x, gyro_y, gyro_z;
        float accel_x, accel_y, accel_z;
        float vehicle_heading;
        float total_acceleration;
        int message_id;
        float uptime_seconds;
        std::string timestamp;
    };

    SensorData data;

    std::unique_ptr<cJSON, decltype(&cJSON_Delete)> toJSON() const {
        std::unique_ptr<cJSON, decltype(&cJSON_Delete)> json(cJSON_CreateObject(), cJSON_Delete);
        
        cJSON_AddStringToObject(json.get(), "timestamp", data.timestamp.c_str());
        cJSON_AddNumberToObject(json.get(), "speed_ms", std::round(data.speed_ms * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "voltage_v", std::round(data.voltage_v * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "current_a", std::round(data.current_a * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "power_w", std::round(data.power_w * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "energy_j", std::round(data.energy_j * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "distance_m", std::round(data.distance_m * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "latitude", std::round(data.latitude * 1000000) / 1000000);
        cJSON_AddNumberToObject(json.get(), "longitude", std::round(data.longitude * 1000000) / 1000000);
        cJSON_AddNumberToObject(json.get(), "gyro_x", std::round(data.gyro_x * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "gyro_y", std::round(data.gyro_y * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "gyro_z", std::round(data.gyro_z * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "accel_x", std::round(data.accel_x * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "accel_y", std::round(data.accel_y * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "accel_z", std::round(data.accel_z * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "vehicle_heading", std::round(std::fmod(data.vehicle_heading, 360.0f) * 100) / 100);
        cJSON_AddNumberToObject(json.get(), "total_acceleration", std::round(data.total_acceleration * 1000) / 1000);
        cJSON_AddNumberToObject(json.get(), "message_id", data.message_id);
        cJSON_AddNumberToObject(json.get(), "uptime_seconds", std::round(data.uptime_seconds * 100) / 100);
        
        return json;
    }
};

class RandomGenerator {
private:
    std::mt19937 gen;
    std::normal_distribution<float> normal_dist;
    
public:
    RandomGenerator() : gen(std::time(nullptr)), normal_dist(0.0f, 1.0f) {}
    
    float gaussian(float mean, float stddev) {
        return normal_dist(gen) * stddev + mean;
    }
    
    float uniform(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);
    }
};

class TelemetrySimulator {
private:
    float simulation_time;
    float cumulative_energy;
    float cumulative_distance;
    float vehicle_heading;
    float prev_speed;
    int message_count;
    int64_t start_time_us;
    RandomGenerator rng;
    
    static constexpr float clamp(float value, float min_val, float max_val) {
        return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
    }
    
    std::string getISOTimestamp() const {
        struct timeval tv;
        struct tm timeinfo;
        
        gettimeofday(&tv, nullptr);
        gmtime_r(&tv.tv_sec, &timeinfo);
        
        std::ostringstream oss;
        oss << std::setfill('0') 
            << std::setw(4) << (timeinfo.tm_year + 1900) << "-"
            << std::setw(2) << (timeinfo.tm_mon + 1) << "-"
            << std::setw(2) << timeinfo.tm_mday << "T"
            << std::setw(2) << timeinfo.tm_hour << ":"
            << std::setw(2) << timeinfo.tm_min << ":"
            << std::setw(2) << timeinfo.tm_sec << "."
            << std::setw(3) << (tv.tv_usec / 1000) << "Z";
        
        return oss.str();
    }
    
public:
    TelemetrySimulator() : simulation_time(0.0f), cumulative_energy(0.0f), 
                          cumulative_distance(0.0f), vehicle_heading(0.0f),
                          prev_speed(0.0f), message_count(0), 
                          start_time_us(esp_timer_get_time()) {}
    
    TelemetryData generateTelemetryData() {
        TelemetryData telemetry;
        telemetry.data.timestamp = getISOTimestamp();
        
        // Generate realistic speed (0-25 m/s with variations)
        float base_speed = 15.0f + 5.0f * std::sin(simulation_time * 0.1f);
        float speed_variation = rng.gaussian(0, 1.5f);
        float speed = clamp(base_speed + speed_variation, 0.0f, 25.0f);
        
        // Generate electrical system data
        float voltage = clamp(48.0f + rng.gaussian(0, 1.5f), 40.0f, 55.0f);
        float current = clamp(8.0f + speed * 0.2f + rng.gaussian(0, 1.0f), 0.0f, 15.0f);
        float power = voltage * current;
        
        // Accumulate energy and distance
        float energy_delta = power * (TelemetryConfig::PUBLISH_INTERVAL / 1000.0f);
        float distance_delta = speed * (TelemetryConfig::PUBLISH_INTERVAL / 1000.0f);
        
        cumulative_energy += energy_delta;
        cumulative_distance += distance_delta;
        
        // Generate GPS coordinates (simulated route)
        constexpr float base_lat = 40.7128f;
        constexpr float base_lon = -74.0060f;
        float lat_offset = 0.001f * std::sin(simulation_time * 0.05f);
        float lon_offset = 0.001f * std::cos(simulation_time * 0.05f);
        
        double latitude = base_lat + lat_offset + rng.gaussian(0, 0.0001f);
        double longitude = base_lon + lon_offset + rng.gaussian(0, 0.0001f);
        
        // Generate Gyroscope data (angular velocity in deg/s)
        float turning_rate = 2.0f * std::sin(simulation_time * 0.08f);
        float gyro_x = rng.gaussian(0, 0.5f);
        float gyro_y = rng.gaussian(0, 0.3f);
        float gyro_z = turning_rate + rng.gaussian(0, 0.8f);
        
        // Update vehicle heading
        vehicle_heading += gyro_z * (TelemetryConfig::PUBLISH_INTERVAL / 1000.0f);
        
        // Generate Accelerometer data (m/s2)
        float speed_acceleration = (speed - prev_speed) / (TelemetryConfig::PUBLISH_INTERVAL / 1000.0f);
        prev_speed = speed;
        
        float accel_x = speed_acceleration + rng.gaussian(0, 0.2f);
        float accel_y = turning_rate * speed * 0.1f + rng.gaussian(0, 0.1f);
        float accel_z = 9.81f + rng.gaussian(0, 0.05f);
        
        // Add vibrations correlated with speed
        float vibration_factor = speed * 0.02f;
        accel_x += rng.gaussian(0, vibration_factor);
        accel_y += rng.gaussian(0, vibration_factor);
        accel_z += rng.gaussian(0, vibration_factor);
        
        // Calculate total acceleration
        float total_acceleration = std::sqrt(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z);
        
        // Get uptime in seconds
        int64_t current_time_us = esp_timer_get_time();
        float uptime_seconds = (current_time_us - start_time_us) / 1000000.0f;
        
        simulation_time += 1.0f;
        message_count++;
        
        // Fill telemetry data
        telemetry.data.speed_ms = speed;
        telemetry.data.voltage_v = voltage;
        telemetry.data.current_a = current;
        telemetry.data.power_w = power;
        telemetry.data.energy_j = cumulative_energy;
        telemetry.data.distance_m = cumulative_distance;
        telemetry.data.latitude = latitude;
        telemetry.data.longitude = longitude;
        telemetry.data.gyro_x = gyro_x;
        telemetry.data.gyro_y = gyro_y;
        telemetry.data.gyro_z = gyro_z;
        telemetry.data.accel_x = accel_x;
        telemetry.data.accel_y = accel_y;
        telemetry.data.accel_z = accel_z;
        telemetry.data.vehicle_heading = vehicle_heading;
        telemetry.data.total_acceleration = total_acceleration;
        telemetry.data.message_id = message_count;
        telemetry.data.uptime_seconds = uptime_seconds;
        
        return telemetry;
    }
};

class MQTTClient {
private:
    esp_mqtt_client_handle_t client;
    EventGroupHandle_t event_group;
    static constexpr int CONNECTED_BIT = BIT0;
    static const char* TAG;
    
    std::string generateClientId() {
        uint8_t mac[6];
        esp_wifi_get_mac(WIFI_IF_STA, mac);
        
        std::ostringstream oss;
        oss << TelemetryConfig::ABLY_CLIENT_ID_PREFIX 
            << std::hex << std::setfill('0');
        for (int i = 0; i < 6; ++i) {
            oss << std::setw(2) << static_cast<unsigned>(mac[i]);
        }
        
        return oss.str();
    }
    
    static void mqttEventHandler(void *handler_args, esp_event_base_t base, 
                                int32_t event_id, void *event_data) {
        auto* instance = static_cast<MQTTClient*>(handler_args);
        instance->handleMqttEvent(base, event_id, event_data);
    }
    
    void handleMqttEvent(esp_event_base_t base, int32_t event_id, void *event_data) {
        auto event = static_cast<esp_mqtt_event_handle_t>(event_data);
        
        switch (static_cast<esp_mqtt_event_id_t>(event_id)) {
            case MQTT_EVENT_CONNECTED:
                ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
                xEventGroupSetBits(event_group, CONNECTED_BIT);
                break;
                
            case MQTT_EVENT_DISCONNECTED:
                ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
                xEventGroupClearBits(event_group, CONNECTED_BIT);
                break;
                
            case MQTT_EVENT_PUBLISHED:
                ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
                break;
                
            case MQTT_EVENT_ERROR:
                ESP_LOGE(TAG, "MQTT_EVENT_ERROR");
                if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                    ESP_LOGE(TAG, "Last errno string (%s)", 
                            strerror(event->error_handle->esp_transport_sock_errno));
                }
                break;
                
            default:
                ESP_LOGI(TAG, "Other event id:%d", event->event_id);
                break;
        }
    }
    
public:
    MQTTClient() : client(nullptr) {
        event_group = xEventGroupCreate();
    }
    
    ~MQTTClient() {
        if (client) {
            esp_mqtt_client_stop(client);
            esp_mqtt_client_destroy(client);
        }
        if (event_group) {
            vEventGroupDelete(event_group);
        }
    }
    
    bool initialize() {
        std::string client_id = generateClientId();
        ESP_LOGI(TAG, "Using client ID: %s", client_id.c_str());
        
        esp_mqtt_client_config_t mqtt_cfg = {};
        mqtt_cfg.broker.address.hostname = TelemetryConfig::MQTT_BROKER_HOST;
        mqtt_cfg.broker.address.port = TelemetryConfig::MQTT_BROKER_PORT;
        mqtt_cfg.broker.address.transport = MQTT_TRANSPORT_OVER_SSL;
        mqtt_cfg.broker.verification.skip_cert_common_name_check = true;
        mqtt_cfg.broker.verification.crt_bundle_attach = esp_crt_bundle_attach;
        mqtt_cfg.credentials.username = TelemetryConfig::MQTT_USERNAME;
        mqtt_cfg.credentials.authentication.password = TelemetryConfig::MQTT_PASSWORD;
        mqtt_cfg.credentials.client_id = client_id.c_str();
        mqtt_cfg.session.keepalive = 60;
        mqtt_cfg.session.disable_clean_session = false;
        mqtt_cfg.network.timeout_ms = 10000;
        mqtt_cfg.network.refresh_connection_after_ms = 20000;
        mqtt_cfg.network.disable_auto_reconnect = false;
        
        client = esp_mqtt_client_init(&mqtt_cfg);
        if (!client) {
            ESP_LOGE(TAG, "Failed to initialize MQTT client");
            return false;
        }
        
        esp_mqtt_client_register_event(client, MQTT_EVENT_ANY, mqttEventHandler, this);
        
        esp_err_t result = esp_mqtt_client_start(client);
        if (result != ESP_OK) {
            ESP_LOGE(TAG, "Failed to start MQTT client: %s", esp_err_to_name(result));
            return false;
        }
        
        return true;
    }
    
    bool waitForConnection(uint32_t timeout_ms = 30000) {
        EventBits_t bits = xEventGroupWaitBits(event_group, CONNECTED_BIT, 
                                              pdFALSE, pdFALSE, pdMS_TO_TICKS(timeout_ms));
        return (bits & CONNECTED_BIT) != 0;
    }
    
    bool publish(const TelemetryData& telemetry) {
        if (!client) {
            ESP_LOGE(TAG, "MQTT client not initialized");
            return false;
        }
        
        EventBits_t bits = xEventGroupWaitBits(event_group, CONNECTED_BIT, 
                                              pdFALSE, pdFALSE, pdMS_TO_TICKS(2000));
        
        if (!(bits & CONNECTED_BIT)) {
            ESP_LOGW(TAG, "MQTT not connected, skipping publish");
            return false;
        }
        
        auto json = telemetry.toJSON();
        std::unique_ptr<char, decltype(&free)> json_string(cJSON_Print(json.get()), free);
        
        if (!json_string) {
            ESP_LOGE(TAG, "Failed to create JSON string");
            return false;
        }
        
        int msg_id = esp_mqtt_client_publish(client, TelemetryConfig::ABLY_CHANNEL, 
                                           json_string.get(), 0, 0, 0);
        
        if (msg_id >= 0) {
            ESP_LOGI(TAG, "Telemetry published successfully, msg_id=%d", msg_id);
            ESP_LOGI(TAG, "Published to topic: %s", TelemetryConfig::ABLY_CHANNEL);
            return true;
        } else {
            ESP_LOGE(TAG, "Failed to publish telemetry data");
            return false;
        }
    }
};

const char* MQTTClient::TAG = "MQTT_CLIENT";

class WiFiManager {
private:
    EventGroupHandle_t event_group;
    static constexpr int CONNECTED_BIT = BIT0;
    static const char* TAG;
    
    static void eventHandler(void* arg, esp_event_base_t event_base,
                           int32_t event_id, void* event_data) {
        auto* instance = static_cast<WiFiManager*>(arg);
        instance->handleWiFiEvent(event_base, event_id, event_data);
    }
    
    void handleWiFiEvent(esp_event_base_t event_base, int32_t event_id, void* event_data) {
        if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
            esp_wifi_connect();
        } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
            esp_wifi_connect();
            ESP_LOGI(TAG, "retry to connect to the AP");
            xEventGroupClearBits(event_group, CONNECTED_BIT);
        } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
            auto event = static_cast<ip_event_got_ip_t*>(event_data);
            ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
            xEventGroupSetBits(event_group, CONNECTED_BIT);
        }
    }
    
public:
    WiFiManager() {
        event_group = xEventGroupCreate();
    }
    
    ~WiFiManager() {
        if (event_group) {
            vEventGroupDelete(event_group);
        }
    }
    
    bool initialize() {
        ESP_ERROR_CHECK(esp_netif_init());
        ESP_ERROR_CHECK(esp_event_loop_create_default());
        esp_netif_create_default_wifi_sta();
        
        wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
        ESP_ERROR_CHECK(esp_wifi_init(&cfg));
        
        esp_event_handler_instance_t instance_any_id;
        esp_event_handler_instance_t instance_got_ip;
        ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                            ESP_EVENT_ANY_ID,
                                                            &eventHandler,
                                                            this,
                                                            &instance_any_id));
        ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                            IP_EVENT_STA_GOT_IP,
                                                            &eventHandler,
                                                            this,
                                                            &instance_got_ip));
        
        wifi_config_t wifi_config = {};
        strcpy(reinterpret_cast<char*>(wifi_config.sta.ssid), TelemetryConfig::WIFI_SSID);
        strcpy(reinterpret_cast<char*>(wifi_config.sta.password), TelemetryConfig::WIFI_PASSWORD);
        wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
        
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
        ESP_ERROR_CHECK(esp_wifi_start());
        
        ESP_LOGI(TAG, "WiFi initialization finished.");
        return true;
    }
    
    bool waitForConnection() {
        ESP_LOGI(TAG, "Waiting for WiFi connection...");
        xEventGroupWaitBits(event_group, CONNECTED_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
        ESP_LOGI(TAG, "WiFi connected!");
        return true;
    }
};

const char* WiFiManager::TAG = "WIFI_MANAGER";

class TelemetrySystem {
private:
    static const char* TAG;
    WiFiManager wifi_manager;
    MQTTClient mqtt_client;
    TelemetrySimulator simulator;
    
public:
    void run() {
        ESP_LOGI(TAG, "Initializing ESP32 Telemetry Simulator with MQTT over SSL");
        
        // Initialize WiFi
        wifi_manager.initialize();
        wifi_manager.waitForConnection();
        
        // Initialize MQTT
        mqtt_client.initialize();
        if (mqtt_client.waitForConnection()) {
            ESP_LOGI(TAG, "MQTT connected!");
        } else {
            ESP_LOGE(TAG, "MQTT connection timeout! Continuing anyway...");
        }
        
        ESP_LOGI(TAG, "Starting telemetry simulation...");
        
        while (true) {
            // Generate telemetry data
            TelemetryData telemetry = simulator.generateTelemetryData();
            
            // Send via MQTT
            mqtt_client.publish(telemetry);
            
            // Log telemetry data locally
            auto json = telemetry.toJSON();
            std::unique_ptr<char, decltype(&free)> json_string(cJSON_Print(json.get()), free);
            ESP_LOGI(TAG, "Generated telemetry: %s", json_string.get());
            
            // Wait for next iteration
            vTaskDelay(pdMS_TO_TICKS(TelemetryConfig::PUBLISH_INTERVAL));
        }
    }
};

const char* TelemetrySystem::TAG = "TELEMETRY_SYSTEM";

// Task wrapper for C++ class
extern "C" void telemetry_task(void *pvParameters) {
    auto* system = static_cast<TelemetrySystem*>(pvParameters);
    system->run();
}

extern "C" void app_main(void) {
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Create telemetry system instance
    static TelemetrySystem telemetry_system;
    
    // Create telemetry task
    xTaskCreate(telemetry_task, "telemetry_task", 8192, &telemetry_system, 5, nullptr);
}
