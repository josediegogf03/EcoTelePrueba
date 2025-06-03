import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import time
import logging
from collections import deque

from ably import AblyRealtime, AblyException

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Ably Configuration ---
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
ABLY_API_KEY = os.environ.get("ABLY_API_KEY", ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"  # Must match maindata.py

# --- Constants ---
MAX_DATAPOINTS_IN_DASHBOARD = 500  # Max points to keep in the dashboard display
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
    page_title="Shell Eco-marathon Telemetry Dashboard V5 (Ably Realtime)",
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
</style>
""",
    unsafe_allow_html=True,
)


# --- Charting and KPI Functions ---
def calculate_kpis(df):
    if df.empty or not all(
        col in df.columns for col in ["energy_j", "speed_ms", "distance_m", "power_w"]
    ):
        return {
            k: 0
            for k in [
                "total_energy_mj",
                "max_speed_ms",
                "avg_speed_ms",
                "total_distance_km",
                "avg_power_w",
                "efficiency_km_per_mj",
            ]
        }
    for col in ["energy_j", "speed_ms", "distance_m", "power_w"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["energy_j", "speed_ms", "distance_m", "power_w"])
    if df.empty:
        return {
            k: 0
            for k in [
                "total_energy_mj",
                "max_speed_ms",
                "avg_speed_ms",
                "total_distance_km",
                "avg_power_w",
                "efficiency_km_per_mj",
            ]
        }
    total_energy = df["energy_j"].sum() / 1000000
    max_speed = df["speed_ms"].max()
    avg_speed = df["speed_ms"].mean()
    total_distance = df["distance_m"].max() / 1000
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


def create_speed_chart(df):
    if (
        df.empty
        or "timestamp" not in df.columns
        or "speed_ms" not in df.columns
    ):
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No data)", height=400
        )
    df["speed_ms"] = pd.to_numeric(df["speed_ms"], errors="coerce")
    return (
        px.line(
            df.dropna(subset=["timestamp", "speed_ms"]),
            x="timestamp",
            y="speed_ms",
            title="Vehicle Speed Over Time",
            labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        )
        .update_layout(height=400)
    )


def create_power_chart(df):
    if df.empty or not all(
        col in df.columns for col in ["timestamp", "voltage_v", "current_a"]
    ):
        return go.Figure().update_layout(
            title="Electrical Parameters (No data)", height=400
        )
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
    df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df_valid = df.dropna(subset=["timestamp", "voltage_v", "current_a"])
    fig.add_trace(
        go.Scatter(
            x=df_valid["timestamp"], y=df_valid["voltage_v"], name="Voltage (V)"
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_valid["timestamp"], y=df_valid["current_a"], name="Current (A)"
        ),
        secondary_y=True,
    )
    return fig.update_layout(
        title_text="Electrical Parameters Over Time",
        height=400,
        xaxis_title="Time",
        yaxis_title="Voltage (V)",
        yaxis2_title="Current (A)",
    )


def create_efficiency_chart(df):
    if df.empty or not all(
        col in df.columns
        for col in ["distance_m", "energy_j", "speed_ms", "power_w", "voltage_v"]
    ):
        return go.Figure().update_layout(
            title="Efficiency Analysis (No data)", height=400
        )
    df_copy = df.copy()
    for col in ["distance_m", "energy_j", "speed_ms", "power_w", "voltage_v"]:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    df_copy.dropna(
        subset=["distance_m", "energy_j", "speed_ms", "power_w", "voltage_v"],
        inplace=True,
    )
    if df_copy.empty:
        return go.Figure().update_layout(
            title="Efficiency Analysis (Insufficient data)", height=400
        )
    df_copy["efficiency"] = (
        df_copy["distance_m"] / (df_copy["energy_j"] / 1000000)
    ).replace([np.inf, -np.inf], 0)
    return (
        px.scatter(
            df_copy,
            x="speed_ms",
            y="efficiency",
            color="power_w",
            size="voltage_v",
            title="Efficiency Analysis",
            labels={"speed_ms": "Speed (m/s)", "efficiency": "Efficiency (m/MJ)"},
        )
        .update_layout(height=400)
    )


def create_gps_map(df):
    if df.empty or not all(
        col in df.columns
        for col in ["latitude", "longitude", "speed_ms", "power_w"]
    ):
        return go.Figure().update_layout(
            title="GPS Track (No data)", mapbox_style="open-street-map", height=400
        )
    for col in ["latitude", "longitude", "speed_ms", "power_w"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().update_layout(
            title="GPS Track (Insufficient data)",
            mapbox_style="open-street-map",
            height=400,
        )
    return (
        px.scatter_mapbox(
            df_valid,
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
            mapbox_center=(
                {
                    "lat": df_valid["latitude"].mean(),
                    "lon": df_valid["longitude"].mean(),
                }
                if not df_valid.empty
                else None
            ),
            mapbox_zoom=12 if not df_valid.empty else 1,
        )
    )


# --- Ably Client and Streamlit State Management ---
def get_ably_realtime_client():
    if not ABLY_API_KEY or (
        ABLY_API_KEY == ABLY_API_KEY_FALLBACK
        and ABLY_API_KEY_FALLBACK == "YOUR_PLACEHOLDER_KEY"
    ):
        st.error(
            "Ably API Key is missing or is a placeholder. Please set ABLY_API_KEY environment variable or update script."
        )
        return None
    try:
        # Pass the key directly to AblyRealtime with additional options
        return AblyRealtime(
            key=ABLY_API_KEY, auto_connect=True, log_level=logging.WARNING
        )
    except AblyException as e:
        st.error(f"Error initializing Ably client: {e}")
        return None


# Initialize session state for data and Ably client
if "ably_client" not in st.session_state:
    st.session_state.ably_client = None
if "ably_channel" not in st.session_state:
    st.session_state.ably_channel = None
if "ably_connection_state" not in st.session_state:
    st.session_state.ably_connection_state = "uninitialized"
if "telemetry_data_deque" not in st.session_state:
    st.session_state.telemetry_data_deque = deque(
        maxlen=MAX_DATAPOINTS_IN_DASHBOARD
    )


def message_callback(msg):
    try:
        data_point = msg.data
        # Ensure timestamp is datetime object
        if "timestamp" in data_point:
            data_point["timestamp"] = pd.to_datetime(data_point["timestamp"])

        # Add to deque (which automatically handles maxlen)
        st.session_state.telemetry_data_deque.append(data_point)

        # Trigger a rerun to update the UI
        st.rerun()

    except Exception as e:
        logging.error(
            f"Error processing message from Ably: {e} - Data: {msg.data}"
        )


def subscribe_to_ably_channel():
    if st.session_state.ably_client and st.session_state.ably_channel:
        try:
            st.session_state.ably_channel.subscribe(
                "telemetry_update", message_callback
            )
            logging.info(
                f"Subscribed to Ably channel '{TELEMETRY_CHANNEL_NAME}' for 'telemetry_update' events."
            )
            st.session_state.ably_connection_state = "Subscribed"
        except AblyException as e:
            st.error(f"Error subscribing to Ably channel: {e}")
            st.session_state.ably_connection_state = f"Subscription Error: {e}"
            logging.error(f"Ably subscription error: {e}")
        except Exception as e:
            st.error(f"Unexpected error during Ably subscription: {e}")
            st.session_state.ably_connection_state = (
                f"Unexpected Subscription Error: {e}"
            )
            logging.error(f"Unexpected Ably subscription error: {e}")


def setup_ably_connection():
    if st.session_state.ably_client is None:
        st.session_state.ably_client = get_ably_realtime_client()

    if st.session_state.ably_client and st.session_state.ably_channel is None:

        def on_connection_state_change(state):
            logging.info(f"Ably connection state changed: {state}")
            st.session_state.ably_connection_state = str(state)
            if state == "connected" and st.session_state.ably_channel:
                logging.info("Re-subscribing after Ably re-connection.")
                subscribe_to_ably_channel()

        st.session_state.ably_client.connection.on(
            "update", on_connection_state_change
        )
        st.session_state.ably_client.connection.on(
            "connected", lambda: on_connection_state_change("connected")
        )
        st.session_state.ably_client.connection.on(
            "failed", lambda: on_connection_state_change("failed")
        )
        st.session_state.ably_client.connection.on(
            "closed", lambda: on_connection_state_change("closed")
        )
        st.session_state.ably_client.connection.on(
            "suspended", lambda: on_connection_state_change("suspended")
        )

        st.session_state.ably_channel = st.session_state.ably_client.channels.get(
            TELEMETRY_CHANNEL_NAME
        )
        subscribe_to_ably_channel()


# --- Main Dashboard Logic ---
def main():
    st.markdown(
        f'<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard V5 (Ably Realtime)</h1>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("Connection Status")
    if not ABLY_API_KEY or (
        ABLY_API_KEY == ABLY_API_KEY_FALLBACK
        and ABLY_API_KEY_FALLBACK == "YOUR_PLACEHOLDER_KEY"
    ):
        st.sidebar.error("Ably API Key MISSING!")
    else:
        st.sidebar.info(f"Ably Key: {ABLY_API_KEY[:10]}...")

    st.sidebar.info(f"Ably Channel: {TELEMETRY_CHANNEL_NAME}")
    st.sidebar.write(
        f"Ably Connection: **{st.session_state.get('ably_connection_state', 'Uninitialized')}**"
    )

    # Initialize Ably connection and subscription if not already done
    if (
        st.session_state.ably_client is None
        or st.session_state.ably_channel is None
    ):
        with st.spinner("Connecting to Ably real-time service..."):
            setup_ably_connection()
            if st.session_state.ably_connection_state == "uninitialized":
                time.sleep(1)
                st.rerun()

    # Convert deque to DataFrame for display
    if st.session_state.telemetry_data_deque:
        current_display_df = pd.DataFrame(
            list(st.session_state.telemetry_data_deque)
        )
        current_display_df = current_display_df.reindex(columns=PLACEHOLDER_COLS)
        if "timestamp" in current_display_df.columns:
            current_display_df["timestamp"] = pd.to_datetime(
                current_display_df["timestamp"]
            )
    else:
        current_display_df = pd.DataFrame(columns=PLACEHOLDER_COLS)

    if current_display_df.empty:
        st.warning(
            f"No data received yet from Ably channel '{TELEMETRY_CHANNEL_NAME}'. Waiting for messages..."
        )
        st.info(
            "Ensure `maindata.py` (Ably publisher) is running and publishing data."
        )

    # --- KPIs and Charts ---
    kpis = calculate_kpis(current_display_df.copy())
    st.subheader("📊 Key Performance Indicators")
    cols_kpi = st.columns(6)

    kpi_metrics = [
        ("Total Distance", f"{kpis.get('total_distance_km', 0):.2f} km"),
        ("Max Speed", f"{kpis.get('max_speed_ms', 0):.1f} m/s"),
        ("Avg Speed", f"{kpis.get('avg_speed_ms', 0):.1f} m/s"),
        ("Total Energy", f"{kpis.get('total_energy_mj', 0):.2f} MJ"),
        ("Avg Power", f"{kpis.get('avg_power_w', 0):.1f} W"),
        ("Efficiency", f"{kpis.get('efficiency_km_per_mj', 0):.2f} km/MJ"),
    ]
    for i, (label, value) in enumerate(kpi_metrics):
        with cols_kpi[i]:
            st.metric(label, value)

    st.subheader("📈 Telemetry Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"]
    )

    display_df_for_charts = current_display_df.copy()

    with tab1:
        st.plotly_chart(
            create_speed_chart(display_df_for_charts), use_container_width=True
        )
    with tab2:
        st.plotly_chart(
            create_power_chart(display_df_for_charts), use_container_width=True
        )
    with tab3:
        st.plotly_chart(
            create_efficiency_chart(display_df_for_charts), use_container_width=True
        )
    with tab4:
        st.plotly_chart(
            create_gps_map(display_df_for_charts), use_container_width=True
        )
    with tab5:
        st.subheader(
            f"Raw Telemetry Data (Last {MAX_DATAPOINTS_IN_DASHBOARD} points)"
        )
        st.dataframe(display_df_for_charts, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'><p>Shell Eco-marathon Telemetry Dashboard V5 | Data via Ably Realtime</p></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    if not ABLY_API_KEY or (
        ABLY_API_KEY == ABLY_API_KEY_FALLBACK
        and not os.environ.get("ABLY_API_KEY")
        and ABLY_API_KEY_FALLBACK == "YOUR_PLACEHOLDER_KEY"
    ):
        st.error(
            "CRITICAL: Ably API Key is missing or is a placeholder. App cannot start. Please set the ABLY_API_KEY environment variable or update the script's fallback key."
        )
    else:
        main()
