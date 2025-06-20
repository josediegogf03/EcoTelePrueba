# Shell Eco-marathon Telemetry Dashboard

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)](https://github.com/your-username/eco-telemetry-dashboard) 
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A **real-time telemetry system** for Shell Eco-marathon vehicles.  
Publish mock (or real) sensor data with a lightweight Python publisher and visualize live KPIs, charts & maps in a Streamlit dashboard—deployed seamlessly on Streamlit Community Cloud.

---

## ✨ Features

- **Low-latency WebSocket streaming** via Ably Realtime  
- **Publisher** (`maindata.py`)  
  - Simulates speed, voltage, current, power, cumulative energy/distance, GPS  
  - Publishes JSON events (`telemetry_update`) every _n_ seconds  
  - Automatic reconnect, graceful shutdown, detailed logging  
- **Dashboard** (`dashboard_#.py`)  
  - Connects to the same Ably channel  
  - Buffers incoming messages thread-safely  
  - Live KPIs: total distance, max/avg speed, energy, power, efficiency  
  - Interactive charts: speed-over-time, electrical subsystem, efficiency scatter, GPS track  
  - Raw data preview & CSV download  
  - Auto-refresh, connection stats, error reporting  

---

## 🏛️ Architecture

```text
┌───────────────┐      WebSocket     ┌────────────────────┐      WebSocket     ┌─────────────────────┐
│  Publisher    │ ──────────────────>│   Ably Realtime    │ ──────────────────>│ Streamlit Dashboard │
│   maindata.py │                    │     (Pub/Sub)      │                    │   dashboard_#.py    │
└───────────────┘                    └────────────────────┘                    └─────────────────────┘
```

1. **maindata.py**  
   - Generates & publishes telemetry events to Ably  
2. **Ably Realtime**  
   - Manages WebSocket connections, message routing  
3. **dashboard_#.py**  
   - Subscribes, queues, and renders data in Streamlit  

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ecotele.git
cd eco-telemetry-dashboard
pip install -r requirements.txt
```

### 2. Run the Publisher

```bash
python maindata.py
```

_Watch the console for connection logs & publish events._

### 3. Launch the Dashboard

#### Local

```bash
streamlit run dashboard_#.py
```

#### Streamlit Community Cloud

1. Push this repo to GitHub  
2. In [Streamlit Cloud](https://streamlit.io/cloud), “New app” → Select repo & `dashboard_#.py`  
3. Deploy and enjoy real-time telemetry

---

## 🗂️ Repository Structure

```
├── maindata.py           # Telemetry publisher  
├── dashboard_#.py        # Streamlit subscriber dashboard  
├── requirements.txt      # Python dependencies  
├── README.md             # Project overview & instructions
├── demo1.py              # Dashboard demo verson. (Fully mock data, lacks connection)
└── LICENSE               # MIT License  
```

---

## 📄 License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for full text.

---

> _This project is in **alpha** stage. Feedback, issues and contributions are very welcome!_  
```
