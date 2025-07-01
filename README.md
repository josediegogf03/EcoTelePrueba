
# 🏎️ Shell Eco-marathon Telemetry Dashboard

[![Status: Beta](https://img.shields.io/badge/status-beta-blue)](https://github.com/ChosF/EcoTele/releases/tag/Dashboard_Beta)  
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A **real-time telemetry system** for Shell Eco-marathon vehicles.  
Publish mock—or real—sensor data (including IMU: gyroscope & accelerometer), then visualize live KPIs, charts, maps & custom graphs in a Streamlit dashboard.

---

## ✨ Features

- **Publisher** (`maindata.py`)  
  - Simulates:
    - Vehicle kinematics: `speed_ms`, `distance_m`  
    - Electrical: `voltage_v`, `current_a`, `power_w`, `energy_j`  
    - GPS: `latitude`, `longitude`  
    - IMU: `gyro_x/y/z`, `accel_x/y/z`, derived `total_acceleration`, `vehicle_heading`  
    - Message metadata: `message_id`, `uptime_seconds`  
  - Publishes JSON under `telemetry_update` every _n_ seconds  
  - Automatic reconnect, SIGINT/SIGTERM handlers, graceful cleanup  
  - Detailed logging (every 10th message summary)

- **Dashboard** (`dashboard_050.py`)  
  - Connects to Ably Realtime, buffers data thread-safely  
  - **Nine tabs**:
    1. **Overview** – high-level KPIs  
    2. **Speed** – time-series speed chart  
    3. **Power** – voltage/current & power  
    4. **IMU** – combined gyro & accel trends  
    5. **IMU Detail** – six-panel X/Y/Z subplots  
    6. **Efficiency** – speed vs power scatter  
    7. **GPS** – map-based track & performance  
    8. **Custom** – on-the-fly chart builder (line, scatter, bar, histogram, heatmap)  
    9. **Data** – raw table + CSV download  
  - **Dynamic Charts**: add, rename, delete custom graphs; correlation heatmap  
  - **Visual Enhancements**: theme-aware CSS, sticky header & tabs, modern buttons & cards  
  - **KPIs Extended**: max acceleration, avg gyro magnitude  
  - Terminal logging of subscriber events  

---

## 🏛️ Architecture

```text
┌───────────────┐      WebSocket     ┌────────────────────┐      WebSocket     ┌─────────────────────┐
│  Publisher    │  ─────────────────>│   Ably Realtime    │  ─────────────────>│ Streamlit Dashboard │
│  maindata.py  │                    │     (Pub/Sub)      │                    │  dashboard_050.py   │
└───────────────┘                    └────────────────────┘                    └─────────────────────┘
```

---

## 🚀 Quickstart


# 1. Clone the repo & checkout Beta
```bash
git clone https://github.com/ChosF/EcoTele.git
cd EcoTele
git fetch --tags
git checkout Dashboard_Beta
```
# 2. Install dependencies
```bash
pip install -r requirements.txt
```
# 3. Run the telemetry publisher
```bash
python maindata.py
```
# 4. In a new terminal, launch the dashboard
```bash
streamlit run dashboard_050.py
```
```

> **Tip:** To deploy on Streamlit Community Cloud, point your app to `dashboard_050.py` in this repo.

---

## 🗂️ Repository Structure

```
EcoTele/
├── maindata.py            # Enhanced telemetry publisher w/ IMU support
├── dashboard_050.py       # Streamlit subscriber dashboard (v0.5 Beta)
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # Project overview & instructions
```

---

## 📄 License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

> _This project is now in **Beta**. We welcome your feedback, issues and contributions!_  
```
