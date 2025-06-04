import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import time
import logging
import asyncio
from collections import deque

# Try to import Ably REST, fall back if not installed
try:
    from ably import AblyRest, AblyException
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    AblyRest = None
    AblyException = Exception

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Ably Configuration ---
ABLY_API_KEY_FALLBACK = (
    "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
)
ABLY_API_KEY = os.environ.get("ABLY_API_KEY", ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"

# --- Constants ---
MAX_DATAPOINTS_IN_DASHBOARD = 500
PLACEHOLDER_COLS = [
    "timestamp",
    "speed_ms",
    "voltage_v",
    "current_a",
    "power_w",
    "energy_j",
    "distance_m",
    "latitude",
    "longitude",
]

# --- Page Configuration ---
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard V5",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
}
.error-card {
    background-color: #ff6b35;
    border: 2px solid #e55100;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: white;
}
.warning-card {
    background-color: #d32f2f;
    border: 2px solid #b71c1c;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: white;
}
.info-card {
    background-color: #1976d2;
    border: 2px solid #0d47a1;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: white;
}
.info-card h4, .info-card h5 {
    color: #e3f2fd;
}
.info-card ul {
    color: #f3e5f5;
}
.warning-card h3 {
    color: #ffebee;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Initialize Session State (Run once) ---
if "initialized" not in st.session_state:
    st.session_state.setdefault("connection_error", None)
    st.session_state.setdefault("use_mock_data", False)
    st.session_state.setdefault("ably_rest", None)
    st.session_state.setdefault("ably_channel", None)
    st.session_state.setdefault("last_fetch_time", None)
    st.session_state.telemetry_data_deque = deque(
        maxlen=MAX_DATAPOINTS_IN_DASHBOARD
    )
    st.session_state.initialized = True

# --- Initialize Ably REST Client once ---
if (
    ABLY_AVAILABLE
    and st.session_state.ably_rest is None
    and st.session_state.connection_error is None
):
    try:
        rest = AblyRest(key=ABLY_API_KEY)
        ch = rest.channels.get(TELEMETRY_CHANNEL_NAME)
        st.session_state.ably_rest = rest
        st.session_state.ably_channel = ch
        logging.info("Initialized AblyRest client and channel.")
    except AblyException as e:
        st.session_state.connection_error = f"Ably REST init failed: {e}"
        logging.error(st.session_state.connection_error)

# --- Async Helper ---
async def fetch_history_async(channel, limit):
    """Async wrapper for channel history"""
    try:
        history = await channel.history(direction="forwards", limit=limit)
        return history
    except Exception as e:
        raise AblyException(f"Failed to fetch history: {e}")

def fetch_ably_data():
    """Fetch data from Ably channel, handling both sync and async cases"""
    if not st.session_state.ably_channel:
        return
    
    try:
        # Try async approach first
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            history = loop.run_until_complete(
                fetch_history_async(
                    st.session_state.ably_channel, 
                    MAX_DATAPOINTS_IN_DASHBOARD
                )
            )
        finally:
            if loop:
                loop.close()
        
        # Clear and refill the deque
        st.session_state.telemetry_data_deque.clear()
        for msg in history.items:
            st.session_state.telemetry_data_deque.append(msg.data)
        
        st.session_state.last_fetch_time = datetime.now()
        
    except Exception as e:
        st.error(f"Error fetching from Ably: {e}")

# --- KPI & Chart Helpers ---
@st.cache_data
def calculate_kpis(records):
    df = pd.DataFrame(records)
    keys = [
        "total_energy_mj",
        "max_speed_ms",
        "avg_speed_ms",
        "total_distance_km",
        "avg_power_w",
        "efficiency_km_per_mj",
    ]
    if df.empty or not all(
        col in df.columns for col in ["energy_j", "speed_ms", "distance_m", "power_w"]
    ):
        return {k: 0 for k in keys}
    for col in ["energy_j", "speed_ms", "distance_m", "power_w"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["energy_j", "speed_ms", "distance_m", "power_w"])
    if df.empty:
        return {k: 0 for k in keys}
    total_energy = df["energy_j"].sum() / 1_000_000
    max_speed = df["speed_ms"].max()
    avg_speed = df["speed_ms"].mean()
    total_distance = df["distance_m"].max() / 1_000
    avg_power = df["power_w"].mean()
    efficiency = total_distance / total_energy if total_energy > 0 else 0
    return {
        "total_energy_mj": total_energy,
        "max_speed_ms": max_speed,
        "avg_speed_ms": avg_speed,
        "total_distance_km": total_distance,
        "avg_power_w": avg_power,
        "efficiency_km_per_mj": efficiency,
    }

@st.cache_data
def create_speed_chart(records):
    df = pd.DataFrame(records)
    if df.empty or "timestamp" not in df.columns or "speed_ms" not in df.columns:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No data)", height=400
        )
    df["speed_ms"] = pd.to_numeric(df["speed_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp", "speed_ms"])
    if df.empty:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No valid data)", height=400
        )
    return (
        px.line(
            df,
            x="timestamp",
            y="speed_ms",
            title="Vehicle Speed Over Time",
            labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        )
        .update_layout(height=400)
    )

@st.cache_data
def create_power_chart(records):
    df = pd.DataFrame(records)
    if df.empty or not all(
        col in df.columns for col in ["timestamp", "voltage_v", "current_a"]
    ):
        return go.Figure().update_layout(
            title="Electrical Parameters (No data)", height=400
        )
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
    df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce")
    df = df.dropna(subset=["timestamp", "voltage_v", "current_a"])
    if df.empty:
        return go.Figure().update_layout(
            title="Electrical Parameters (No valid data)", height=400
        )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["voltage_v"], name="Voltage (V)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["current_a"], name="Current (A)"),
        secondary_y=True,
    )
    return fig.update_layout(
        title_text="Electrical Parameters Over Time",
        height=400,
        xaxis_title="Time",
        yaxis_title="Voltage (V)",
        yaxis2_title="Current (A)",
    )

@st.cache_data
def create_efficiency_chart(records):
    df = pd.DataFrame(records)
    required = ["distance_m", "energy_j", "speed_ms", "power_w", "voltage_v"]
    if df.empty or not all(col in df.columns for col in required):
        return go.Figure().update_layout(
            title="Efficiency Analysis (No data)", height=400
        )
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    if df.empty:
        return go.Figure().update_layout(
            title="Efficiency Analysis (Insufficient data)", height=400
        )
    df["efficiency"] = (
        df["distance_m"] / (df["energy_j"] / 1_000_000)
    ).replace([np.inf, -np.inf], 0)
    return (
        px.scatter(
            df,
            x="speed_ms",
            y="efficiency",
            color="power_w",
            size="voltage_v",
            title="Efficiency Analysis",
            labels={"speed_ms": "Speed (m/s)", "efficiency": "Efficiency (m/MJ)"},
        )
        .update_layout(height=400)
    )

@st.cache_data
def create_gps_map(records):
    df = pd.DataFrame(records)
    if df.empty or not all(
        col in df.columns for col in ["latitude", "longitude", "speed_ms", "power_w"]
    ):
        return go.Figure().update_layout(
            title="GPS Track (No data)",
            mapbox_style="open-street-map",
            height=400,
        )
    for col in ["latitude", "longitude", "speed_ms", "power_w"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        return go.Figure().update_layout(
            title="GPS Track (Insufficient data)",
            mapbox_style="open-street-map",
            height=400,
        )
    return (
        px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color="speed_ms",
            size="power_w",
            mapbox_style="open-street-map",
            title="Vehicle Track",
            height=400,
            zoom=10,
        )
        .update_layout(
            mapbox_center={
                "lat": df["latitude"].mean(),
                "lon": df["longitude"].mean(),
            },
            mapbox_zoom=12,
        )
    )

@st.cache_data
def generate_mock_data(num_points=100):
    import random
    now = datetime.now()
    data = []
    base_speed, base_voltage, base_current = 15, 48, 10
    cum_dist = cum_energy = 0
    for i in range(num_points):
        ts = now - timedelta(seconds=2 * i)
        speed = max(
            0,
            base_speed + random.gauss(0, 2) + np.sin(i * 0.1) * 3
        )
        voltage = base_voltage + random.gauss(0, 1.5)
        current = max(
            0,
            base_current + random.gauss(0, 1.5)
            + (speed - base_speed) * 0.3,
        )
        power = voltage * current
        cum_energy += power * 2
        cum_dist += speed * 2
        lat = 40.7128 + i * 0.0001 + random.gauss(0, 0.0001)
        lon = -74.0060 + i * 0.0001 + random.gauss(0, 0.0001)
        data.append({
            "timestamp": ts,
            "speed_ms": round(speed, 2),
            "voltage_v": round(voltage, 2),
            "current_a": round(current, 2),
            "power_w": round(power, 2),
            "energy_j": round(cum_energy, 2),
            "distance_m": round(cum_dist, 2),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
        })
    return pd.DataFrame(data[::-1])

# --- Sidebar ---
def render_sidebar():
    with st.sidebar:
        st.header("🔧 Configuration")
        st.subheader("Connection Status")
        
        if st.session_state.connection_error:
            st.error(f"❌ Connection Error: {st.session_state.connection_error}")
            st.subheader("📊 Data Source")
            use_mock = st.checkbox(
                "Use Mock Data for Demo",
                value=st.session_state.use_mock_data,
            )
            if use_mock != st.session_state.use_mock_data:
                st.session_state.use_mock_data = use_mock
                st.rerun()
            if use_mock:
                st.info("🎭 Using simulated telemetry data")
            else:
                st.warning("⏸️ No data source available")
        else:
            st.success("✅ Real-time connection available")
            st.info(f"Channel: {TELEMETRY_CHANNEL_NAME}")
            
            # Manual refresh button
            if st.button("🔄 Refresh Data"):
                fetch_ably_data()
                st.rerun()
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)
            if auto_refresh:
                # Check if we need to refresh
                now = datetime.now()
                if (st.session_state.last_fetch_time is None or 
                    (now - st.session_state.last_fetch_time).seconds >= 5):
                    fetch_ably_data()
                    st.rerun()
            
            if st.session_state.last_fetch_time:
                st.caption(f"Last updated: {st.session_state.last_fetch_time.strftime('%H:%M:%S')}")

# --- Main ---
def main():
    st.markdown(
        '<h1 class="main-header">'
        '🏎️ Shell Eco-marathon Telemetry Dashboard V5'
        '</h1>',
        unsafe_allow_html=True,
    )

    render_sidebar()

    # Fetch data on initial load
    if (
        st.session_state.ably_channel
        and not st.session_state.use_mock_data
        and not st.session_state.connection_error
        and st.session_state.last_fetch_time is None
    ):
        fetch_ably_data()

    # --- Build DataFrame ---
    if st.session_state.connection_error and not st.session_state.use_mock_data:
        df = pd.DataFrame(columns=PLACEHOLDER_COLS)
    elif st.session_state.use_mock_data:
        df = generate_mock_data()
        st.info("🎭 Displaying simulated telemetry data")
    else:
        if st.session_state.telemetry_data_deque:
            df = pd.DataFrame(list(st.session_state.telemetry_data_deque))
            df = df.reindex(columns=PLACEHOLDER_COLS)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            df = pd.DataFrame(columns=PLACEHOLDER_COLS)
            st.warning(
                f"⏳ Waiting for data on '{TELEMETRY_CHANNEL_NAME}'"
            )

    records = df.to_dict("records") if not df.empty else []

    # --- KPIs ---
    kpis = calculate_kpis(records)
    st.subheader("📊 Key Performance Indicators")
    cols = st.columns(6)
    metrics = [
        ("Total Distance", f"{kpis['total_distance_km']:.2f} km"),
        ("Max Speed",      f"{kpis['max_speed_ms']:.1f} m/s"),
        ("Avg Speed",      f"{kpis['avg_speed_ms']:.1f} m/s"),
        ("Total Energy",   f"{kpis['total_energy_mj']:.2f} MJ"),
        ("Avg Power",      f"{kpis['avg_power_w']:.1f} W"),
        ("Efficiency",     f"{kpis['efficiency_km_per_mj']:.2f} km/MJ"),
    ]
    for i, (lbl, val) in enumerate(metrics):
        with cols[i]:
            st.metric(lbl, val)

    # --- Charts ---
    st.subheader("📈 Telemetry Analytics")
    t1, t2, t3, t4, t5 = st.tabs(
        ["Speed", "Power", "Efficiency", "GPS", "Raw Data"]
    )
    with t1:
        st.plotly_chart(create_speed_chart(records), use_container_width=True)
    with t2:
        st.plotly_chart(create_power_chart(records), use_container_width=True)
    with t3:
        st.plotly_chart(create_efficiency_chart(records), use_container_width=True)
    with t4:
        st.plotly_chart(create_gps_map(records), use_container_width=True)
    with t5:
        st.subheader(f"Raw Telemetry (Last {MAX_DATAPOINTS_IN_DASHBOARD} pts)")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"telemetry_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;'>"
        "Shell Eco-marathon Telemetry Dashboard V5 | Real-Time via Ably REST"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
