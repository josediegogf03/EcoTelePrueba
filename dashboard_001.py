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
import asyncio
from collections import deque

# Try to import Ably, but handle if it's not available
try:
    from ably import AblyRealtime, AblyException
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    AblyRealtime = None
    AblyException = Exception

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Ably Configuration ---
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
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
    .error-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .info-card {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
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
    if df.empty or "timestamp" not in df.columns or "speed_ms" not in df.columns:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No data)", height=400
        )
    df["speed_ms"] = pd.to_numeric(df["speed_ms"], errors="coerce")
    return px.line(
        df.dropna(subset=["timestamp", "speed_ms"]),
        x="timestamp",
        y="speed_ms",
        title="Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
    ).update_layout(height=400)


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
    return px.scatter(
        df_copy,
        x="speed_ms",
        y="efficiency",
        color="power_w",
        size="voltage_v",
        title="Efficiency Analysis",
        labels={"speed_ms": "Speed (m/s)", "efficiency": "Efficiency (m/MJ)"},
    ).update_layout(height=400)


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
    return px.scatter_mapbox(
        df_valid,
        lat="latitude",
        lon="longitude",
        color="speed_ms",
        size="power_w",
        mapbox_style="open-street-map",
        title="Vehicle Track",
        height=400,
        zoom=10,
    ).update_layout(
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


# --- Error Handling and Connection Status ---
def check_event_loop():
    """Check if there's a running event loop"""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def display_connection_issues():
    """Display comprehensive connection status and troubleshooting"""
    st.markdown(
        """
    <div class="warning-card">
        <h3>⚠️ Real-time Connection Issue Detected</h3>
        <p><strong>Issue:</strong> Ably Realtime client cannot establish connection due to event loop constraints in Streamlit environment.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-card">
        <h4>🔧 Technical Details</h4>
        <ul>
            <li><strong>Problem:</strong> Ably Realtime requires an asyncio event loop, but Streamlit runs synchronously</li>
            <li><strong>Error:</strong> "RuntimeError: no running event loop" when trying to create async tasks</li>
            <li><strong>Impact:</strong> Real-time data streaming is not available in this environment</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-card">
        <h4>💡 Recommended Solutions</h4>
        
        <h5>Option 1: Use Ably REST API (Recommended for Streamlit)</h5>
        <ul>
            <li>Switch to polling-based data retrieval using Ably's REST API</li>
            <li>Use <code>st.rerun()</code> with a timer for periodic updates</li>
            <li>More compatible with Streamlit's execution model</li>
        </ul>
        
        <h5>Option 2: Alternative Deployment</h5>
        <ul>
            <li>Deploy using FastAPI + WebSockets + Streamlit components</li>
            <li>Use a separate async service for real-time data handling</li>
            <li>Implement server-sent events (SSE) for real-time updates</li>
        </ul>
        
        <h5>Option 3: Mock Data Mode</h5>
        <ul>
            <li>Generate simulated telemetry data for demonstration</li>
            <li>Useful for testing dashboard functionality</li>
            <li>Enable mock mode in the sidebar</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )


# --- Safe Ably Client Management ---
def get_ably_realtime_client():
    """Safely attempt to create Ably client with comprehensive error handling"""
    if not ABLY_AVAILABLE:
        return None, "Ably library not available"

    if not ABLY_API_KEY or (
        ABLY_API_KEY == ABLY_API_KEY_FALLBACK
        and ABLY_API_KEY_FALLBACK == "YOUR_PLACEHOLDER_KEY"
    ):
        return None, "API key missing or placeholder"

    # Check for event loop
    if not check_event_loop():
        return None, "No event loop available"

    try:
        # Attempt to create client with minimal configuration
        client = AblyRealtime(key=ABLY_API_KEY, auto_connect=False)
        return client, "Success"
    except RuntimeError as e:
        if "no running event loop" in str(e):
            return None, "Event loop error"
        return None, f"Runtime error: {str(e)}"
    except AblyException as e:
        return None, f"Ably error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# --- Mock Data Generation ---
def generate_mock_data():
    """Generate realistic mock telemetry data"""
    import random
    from datetime import datetime, timedelta

    current_time = datetime.now()
    mock_data = []

    for i in range(50):
        timestamp = current_time - timedelta(seconds=i * 2)
        speed = max(0, 15 + random.gauss(0, 3))  # Around 15 m/s with variation
        voltage = 48 + random.gauss(0, 2)  # Around 48V
        current = max(0, 10 + random.gauss(0, 2))  # Around 10A
        power = voltage * current
        energy = power * 2  # 2 seconds per data point
        distance = speed * 2 * (i + 1)  # Cumulative distance

        mock_data.append(
            {
                "timestamp": timestamp,
                "speed_ms": speed,
                "voltage_v": voltage,
                "current_a": current,
                "power_w": power,
                "energy_j": energy,
                "distance_m": distance,
                "latitude": 40.7128 + random.gauss(0, 0.001),  # NYC area
                "longitude": -74.0060 + random.gauss(0, 0.001),
            }
        )

    return pd.DataFrame(mock_data)


# --- Initialize Session State ---
if "ably_client" not in st.session_state:
    st.session_state.ably_client = None
if "ably_channel" not in st.session_state:
    st.session_state.ably_channel = None
if "ably_connection_state" not in st.session_state:
    st.session_state.ably_connection_state = "uninitialized"
if "connection_error" not in st.session_state:
    st.session_state.connection_error = None
if "telemetry_data_deque" not in st.session_state:
    st.session_state.telemetry_data_deque = deque(
        maxlen=MAX_DATAPOINTS_IN_DASHBOARD
    )
if "use_mock_data" not in st.session_state:
    st.session_state.use_mock_data = False


# --- Main Dashboard Logic ---
def main():
    st.markdown(
        f'<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard V5</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar Configuration
    st.sidebar.header("🔧 Configuration")

    # Connection Status
    st.sidebar.subheader("Connection Status")

    # Try to initialize Ably client safely
    if st.session_state.ably_client is None and st.session_state.connection_error is None:
        client, error_msg = get_ably_realtime_client()
        if client:
            st.session_state.ably_client = client
            st.session_state.ably_connection_state = "Connected"
            st.sidebar.success("✅ Ably client initialized")
        else:
            st.session_state.connection_error = error_msg
            st.session_state.ably_connection_state = f"Error: {error_msg}"

    # Display connection status
    if st.session_state.connection_error:
        st.sidebar.error(f"❌ Connection Error: {st.session_state.connection_error}")

        # Mock data option
        st.sidebar.subheader("📊 Data Source")
        use_mock = st.sidebar.checkbox(
            "Use Mock Data for Demo",
            value=st.session_state.use_mock_data,
            help="Generate simulated telemetry data for testing",
        )
        st.session_state.use_mock_data = use_mock

        if use_mock:
            st.sidebar.info("🎭 Using simulated telemetry data")
        else:
            st.sidebar.warning("⏸️ No data source available")

    else:
        st.sidebar.success("✅ Real-time connection available")
        st.sidebar.info(f"Channel: {TELEMETRY_CHANNEL_NAME}")

    # Main content area
    if st.session_state.connection_error and not st.session_state.use_mock_data:
        # Display connection issues and solutions
        display_connection_issues()

        # Show basic dashboard structure with no data
        current_display_df = pd.DataFrame(columns=PLACEHOLDER_COLS)

    elif st.session_state.use_mock_data:
        # Use mock data
        current_display_df = generate_mock_data()
        st.info("🎭 Currently displaying simulated telemetry data for demonstration purposes")

    else:
        # Use real data (if available)
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
            st.warning(
                f"⏳ Waiting for real-time data from channel '{TELEMETRY_CHANNEL_NAME}'"
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
        if not current_display_df.empty:
            st.dataframe(current_display_df, use_container_width=True)
            
            # Download button
            csv = current_display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available to display")

    # Auto-refresh for mock data
    if st.session_state.use_mock_data:
        time.sleep(2)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'><p>Shell Eco-marathon Telemetry Dashboard V5 | Robust Error Handling</p></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
