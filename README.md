# 🏎️ Shell Eco-marathon Telemetry Dashboard

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)  
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/ChosF/EcoTele/releases/tag/Dashboard_Beta)  

A full-stack, real-time and now historical telemetry system for Shell Eco-marathon vehicles.  
From an ESP32-based transmitter through cloud persistence to a Streamlit dashboard, you get live KPIs, charts, maps and replayable past runs.

---

## 🚀 Pipeline Overview

```text
ESP32 Transmitter (Transmiter.cpp)
        └─ MQTT/SSL → Ably MQTT Broker →
Bridge & DB Writer (maindata.py)
   • republishes live → Ably Realtime Pub/Sub
   • batches & stores → Supabase
        └─ Streamlit Dashboard (dashboard_060.py)
           • Real-time view
           • Historical run browser
```

---

## ✨ v0.6 Beta Highlights

- **ESP32 Hardware Support**  
  Connect your on-vehicle ESP32 running the transmitter program to stream real sensor data via secure MQTT.

- **Cloud Persistence & Sessions**  
  All telemetry is batched and saved in Supabase. Each run is tracked as a distinct session, preserving start time, duration and record counts.

- **Historical Data Mode**  
  Browse, select and replay any past session. Automatic pagination handles large datasets seamlessly.

- **Unified Live + Replay**  
  Live streaming and historical replay share the same timeline view with automatic deduplication, so charts never jump or repeat.

- **Enhanced UX for Big Data**  
  Informative spinners during load, “Large Dataset” notices, dual CSV download (full vs sample), and detailed metrics: time span, data rate, memory usage and source breakdown.

---

## 🎯 Features

1. **ESP32 Transmitter** (`Transmiter.cpp`)  
   • FreeRTOS C++ app publishes speed, voltage, power, GPS, IMU via MQTT over SSL.  

2. **Bridge & Database** (`maindata.py`)  
   • Consumes from MQTT or mock generator.  
   • Republishes live events to Ably Realtime.  
   • Batches writes every few seconds to Supabase with session metadata.  

3. **Dashboard** (`dashboard_060.py`)  
   • Real-time + historical mode selection.  
   • Session list & loader with pagination.  
   • Full KPI suite & rich Plotly charts + custom-chart builder.  
   • CSV exports, dataset statistics and responsive theming.  

4. **Legacy & Prototypes**  
   • `dashboard_050.py` (v0.5) – current live-only build  
   • `dashboard_020.py` & `demo_1.py` – prior alpha prototypes  

---

## 🏛️ Architecture Diagram

```text
┌────────────────────────┐
│  ESP32 Transmitter     │
│  (Transmiter.cpp)      │
└──────────┬─────────────┘
           │ MQTT/SSL → Ably MQTT 
┌──────────▼─────────────┐
│ Bridge & DB Writer     │
│ (maindata.py)          │
│ • Live → Ably Realtime │
│ • Batch → Supabase     │
└──────────┬─────────────┘
           │ Pub/Sub & HTTP 
┌──────────▼─────────────┐
│  Streamlit Dashboard   │
│  (dashboard_060.py)    │
│ • Live view            │
│ • Historical browser   │
└────────────────────────┘
```

---

## 🏃 Quickstart

### Prerequisites

- Python 3.8+  
- ESP-IDF toolchain (for `Transmiter.cpp`)  
- `pip install -r requirements.txt`  

### 1. Flash & Run ESP32

```bash
# in ESP-IDF project
idf.py set-target esp32
idf.py menuconfig   # configure Wi-Fi & MQTT
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### 2. Start Bridge & DB Writer

```bash
cd EcoTele
pip install -r requirements.txt
python maindata.py
```

### 3. Launch Dashboard

```bash
streamlit run dashboard_060.py
```

> Deploy on Streamlit Community Cloud by selecting `dashboard_060.py`.

---

## 🗂️ Repository Structure

```
EcoTele/
├── Transmiter.cpp       # ESP32 data transmitter (MQTT/SSL)
├── maindata.py          # Bridge + batch-to-Supabase service
├── dashboard_060.py     # Current Streamlit dashboard (live + historical)
├── dashboard_050.py     # Live-only dashboard (v0.5 Beta)
├── dashboard_020.py     # Legacy dashboard (v0.2 Alpha)
├── demo_1.py            # First prototype (mock-only)
├── requirements.txt     # Python dependencies
├── LICENSE              # Apache 2.0 License
└── README.md            # This file
```

---

## 🚧 Roadmap & Future Work
  
- Real-sensor integration for powertrain & IMU  

---

## 📄 License

Licensed under the **Apache License 2.0**.  
See [LICENSE](LICENSE) for details.

---

> Feedback, bug reports & contributions:  
> https://github.com/ChosF/EcoTele/issues
