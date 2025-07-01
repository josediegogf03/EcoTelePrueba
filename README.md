# Shell Eco-marathon Telemetry Dashboard

[![Status: Beta](https://img.shields.io/badge/status-beta-blue)](https://github.com/ChosF/EcoTele/releases/tag/Dashboard_Beta)  
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A **real-time telemetry system** for Shell Eco-marathon vehicles.  
Publish mock (or real) sensor data with a lightweight Python publisher and visualize live KPIs, interactive charts & maps in a Streamlit dashboard—deployable on Streamlit Community Cloud.

---

## ✨ v0.5 Beta Highlights

- **Custom Chart Builder**  
  Add, configure and remove your own charts on-the-fly: line, scatter, bar, histogram or heatmap—no code edits required.  
- **Extended Sensor Suite**  
  Ingest gyroscope & accelerometer streams alongside speed, voltage, power and GPS for deeper vehicle insights.  
- **Visual & UX Overhaul**  
  Theme-aware (light/dark) CSS, sticky header, modern buttons, info cards, tooltips and responsive layouts.  

---

## ✨ Features

- Publisher (`maindata.py`):  
  - Simulates & publishes JSON events (`telemetry_update`) over Ably Realtime  
  - Sensors: speed, voltage, current, power, cumulative energy/distance, GPS, IMU (gyro & accel)  
  - Auto-reconnect, graceful shutdown, detailed logging  
- Dashboard (`dashboard_050.py`):  
  - Subscribes to Ably channel, thread-safe message queue  
  - Live KPIs: distance, max/avg speed, energy, power, efficiency, max accel, avg gyro  
  - Tabs: Overview, Speed, Power, IMU, IMU Detail, Efficiency, GPS, **Custom**, Data  
  - Custom chart builder with drag-and-drop controls  
  - Raw data preview & CSV download  
  - Auto-refresh, connection stats & error reporting

---

## 🏛️ Architecture

```text
┌───────────────┐      WebSocket     ┌────────────────────┐      WebSocket     ┌─────────────────────┐
│  Publisher    │ ──────────────────>│   Ably Realtime    │ ──────────────────>│ Streamlit Dashboard │
│   maindata.py │                    │     (Pub/Sub)      │                    │  dashboard_050.py   │
└───────────────┘                    └────────────────────┘                    └─────────────────────┘
```

1. **maindata.py**  
   Generates & publishes telemetry events.  
2. **Ably Realtime**  
   Manages WebSocket connections & message routing.  
3. **dashboard_050.py**  
   Subscribes, buffers and renders data via Streamlit.

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/ChosF/EcoTele.git
cd EcoTele

# Checkout the v0.5 Beta tag
git fetch --tags
git checkout Dashboard_Beta

# Install dependencies
pip install -r requirements.txt
```

1. **Run the Publisher**  
   ```bash
   python maindata.py
   ```
2. **Launch the Dashboard (local)**  
   ```bash
   streamlit run dashboard_050.py
   ```
3. **Deploy on Streamlit Cloud**  
   - Push this repo to GitHub  
   - In Streamlit Community Cloud, “New app” → select `dashboard_050.py`

---

## 🗂️ Repository Structure

```
├── maindata.py           # Telemetry publisher (v0.5 – IMU & sensor support)
├── dashboard_050.py      # Current Streamlit dashboard (v0.5 Beta)
├── dashboard_020.py      # Legacy dashboard build (v0.2 Alpha)
├── demo_1.py             # First prototype (fully mock data, no Ably)
├── requirements.txt      # Python dependencies
├── README.md             # Project overview & instructions
└── LICENSE               # MIT License
```

---

## 📄 License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

> _This project is in **Beta** stage. Feedback, issues and contributions are very welcome!_
