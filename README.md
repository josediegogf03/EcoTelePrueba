# 🏎️ Shell Eco-marathon Telemetry Dashboard

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)  
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/ChosF/EcoTele/releases/tag/Dashboard_Beta)  

A full-stack, real-time telemetry system for Shell Eco-marathon vehicles.  
From an ESP32-based transmitter all the way through to a Streamlit dashboard—you’ll get live KPIs, charts, maps and fully customizable visualizations.


## 🚀 Pipeline Overview

```text
ESP32 Transmitter (Transmiter.cpp)
        └─ publishes mock (soon: real) sensor data via MQTT → 
Ably MQTT Broker (mqtt.ably.io) →
maindata.py (MQTT → Ably Realtime bridge) →
Ably Realtime Pub/Sub →
dashboard_050.py (Streamlit subscriber)
```

---

## ✨ v0.5 Beta Highlights

- **ESP32 Transmitter** (`Transmiter.cpp`)  
  - Runs on your vehicle’s ESP32 module  
  - Publishes speed, voltage, power, GPS, gyroscope & accelerometer data (mock for now) via MQTT over SSL  
  - Ready to swap in real sensor reads  

- **Aggregator Bridge** (`maindata.py`)  
  - Subscribes to the `EcoTele` MQTT topic  
  - Republishes incoming JSON payloads into an Ably Realtime channel  
  - Handles reconnects, back-pressure, and graceful shutdown  

- **Streamlit Dashboard** (`dashboard_050.py`)  
  - Live KPIs: distance, max/avg speed, energy, power, efficiency, max accel, avg gyro  
  - Nine tabs: Overview, Speed, Power, IMU, IMU Detail, Efficiency, GPS, Custom, Data  
  - **Custom Chart Builder** – add/remove line, scatter, bar, histogram or heatmap on-the-fly  
  - Theme-aware styling, sticky header, modern buttons & responsive layout  
  - High-capacity buffer (up to 50 000 points)  

---

## 🎯 Features

1. **Transmitter**  
   • ESP32 C++ application using ESP-IDF & FreeRTOS  
   • MQTT over SSL to `mqtt.ably.io:8883`  
   • Mock-data generator for vehicle dynamics & IMU  

2. **Aggregator (`maindata.py`)**  
   • MQTT client retrieves ESP32 payloads  
   • Ably Realtime client republishes under `telemetry_update`  
   • Thread-safe queue, asyncio integration, error tracking  

3. **Dashboard (`dashboard_050.py`)**  
   • Streamlit frontend: thread-safe subscription to Ably Realtime  
   • Key Performance Indicators + rich Plotly charts  
   • Custom charts & correlation heatmap  
   • CSV download & raw data view  

---

## 🏛️ Architecture Diagram

```text
┌────────────────────────┐
│  ESP32 Transmitter     │
│  (Transmiter.cpp)      │
└──────────┬─────────────┘
           │ MQTT (SSL)
┌──────────▼─────────────┐
│   Ably MQTT Broker     │
│  (mqtt.ably.io:8883)   │
└──────────┬─────────────┘
           │ MQTT
┌──────────▼─────────────┐
│  maindata.py           │
│  (MQTT → Ably Realtime)│
└──────────┬─────────────┘
           │ Ably Realtime
┌──────────▼─────────────┐
│  dashboard_050.py      │
│  (Streamlit subscriber)│
└────────────────────────┘
```

---

## 🏃 Quickstart

### Prerequisites

- Python 3.8+  
- [ESP-IDF toolchain](https://docs.espressif.com/projects/esp-idf/) (for flashing Transmiter.cpp)  
- `pip install -r requirements.txt`  

### 1. Flash the ESP32

```bash
# From your ESP-IDF project directory:
idf.py set-target esp32
idf.py menuconfig        # confirm MQTT & WiFi settings
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### 2. Start the Aggregator

```bash
cd EcoTele
pip install -r requirements.txt
python maindata.py
```

### 3. Launch the Dashboard

```bash
streamlit run dashboard_050.py
```

> **Deploy** on Streamlit Community Cloud by pointing to `dashboard_050.py` in your GitHub repo.

---

## 🗂️ Repository Structure

```
EcoTele/
├── Transmiter.cpp        # ESP32 mock-data transmitter (ESP-IDF / FreeRTOS)
├── maindata.py           # MQTT → Ably Realtime bridge & aggregator
├── maindata_0.py         # Legacy publisher (mock-only v0.2)
├── dashboard_050.py      # Current Streamlit dashboard (v0.5 Beta)
├── dashboard_020.py      # Legacy dashboard build (v0.2 Alpha)
├── demo_1.py             # First prototype (fully mock data, no Ably)
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

---

## 🚧 Roadmap & Future Work

- **Persistent Storage**  
  Store incoming data so users can reconnect/reload without losing history.  
- **Historical Replay**  
  Enable “time-travel” through past telemetry sessions.  
- **Resilience Improvements**  
  Fix the auto-refresh freeze (observed when left off >4 min) and support offline buffering.  
- **Real Sensor Integration**  
  Swap mock generators for live IMU, GPS & power-train feeds.  

---

## 📄 License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for full details.

---

> _Feedback, bug reports & contributions are very welcome!_  
> https://github.com/ChosF/EcoTele/issues  
