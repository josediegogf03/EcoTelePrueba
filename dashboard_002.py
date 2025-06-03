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
@st.cache_data
def initialize_session_defaults():
    """Initialize default values - cached to prevent re-execution"""
    return {
        "ably_client": None,
        "ably_channel": None,
        "ably_connection_state": "uninitialized",
        "connection_error": None,
        "use_mock_data": False,
    }


# Initialize session state only if not already done
if "initialized" not in st.session_state:
    defaults = initialize_session_defaults()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.initialized = True
    st.session_state.telemetry_data_deque = deque(maxlen=MAX_DATAPOINTS_IN_DASHBOARD)


# --- Charting and KPI Functions (Cached for efficiency) ---
@st.cache_data
def calculate_kpis(df_hash):
    """Calculate KPIs - cached based on data hash"""
    df = pd.DataFrame(df_hash)
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


@st.cache_data
def create_speed_chart(df_hash):
    """Create speed chart - cached based on data hash"""
    df = pd.DataFrame(df_hash)
    if df.empty or "timestamp" not in df.columns or "speed_ms" not in df.columns:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No data)", height=400
        )
    
    df["speed_ms"] = pd.to_numeric(df["speed_ms"], errors="coerce")
    df_clean = df.dropna(subset=["timestamp", "speed_ms"])
    
    if df_clean.empty:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No valid data)", height=400
        )
    
    return px.line(
        df_clean,
        x="timestamp",
        y="speed_ms",
        title="Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
    ).update_layout(height=400)


@st.cache_data
def create_power_chart(df_hash):
    """Create power chart - cached based on data hash"""
    df = pd.DataFrame(df_hash)
    if df.empty or not all(
        col in df.columns for col in ["timestamp", "voltage_v", "current_a"]
    ):
        return go.Figure().update_layout(
            title="Electrical Parameters (No data)", height=400
        )
    
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
    df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce")
    df_valid = df.dropna(subset=["timestamp", "voltage_v", "current_a"])
    
    if df_valid.empty:
        return go.Figure().update_layout(
            title="Electrical Parameters (No valid data)", height=400
        )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
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


@st.cache_data
def create_efficiency_chart(df_hash):
    """Create efficiency chart - cached based on data hash"""
    df = pd.DataFrame(df_hash)
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


@st.cache_data
def create_gps_map(df_hash):
    """Create GPS map - cached based on data hash"""
    df = pd.DataFrame(df_hash)
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
        mapbox_center={
            "lat": df_valid["latitude"].mean(),
            "lon": df_valid["longitude"].mean(),
        },
        mapbox_zoom=12,
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
@st.cache_data
def get_ably_connection_status():
    """Check Ably connection status - cached to avoid repeated attempts"""
    if not ABLY_AVAILABLE:
        return None, "Ably library not available"

    if not ABLY_API_KEY or (
        ABLY_API_KEY == ABLY_API_KEY_FALLBACK
        and ABLY_API_KEY_FALLBACK == "YOUR_PLACEHOLDER_KEY"
    ):
        return None, "API key missing or placeholder"

    if not check_event_loop():
        return None, "No event loop available"

    return "available", "Connection possible"


# --- Mock Data Generation ---
@st.cache_data
def generate_mock_data(num_points=100):
    """Generate realistic mock telemetry data - cached for efficiency"""
    import random
    
    current_time = datetime.now()
    mock_data = []
    
    # Generate more realistic data with trends
    base_speed = 15
    base_voltage = 48
    base_current = 10
    cumulative_distance = 0
    cumulative_energy = 0

    for i in range(num_points):
        timestamp = current_time - timedelta(seconds=i * 2)
        
        # Add some realistic variation and trends
        speed_variation = random.gauss(0, 2) + np.sin(i * 0.1) * 3
        speed = max(0, base_speed + speed_variation)
        
        voltage = base_voltage + random.gauss(0, 1.5)
        current = max(0, base_current + random.gauss(0, 1.5) + (speed - base_speed) * 0.3)
        power = voltage * current
        energy_increment = power * 2  # 2 seconds per data point
        cumulative_energy += energy_increment
        
        distance_increment = speed * 2  # 2 seconds per data point
        cumulative_distance += distance_increment

        # GPS coordinates with realistic movement
        lat_base = 40.7128 + (i * 0.0001)  # Moving north
        lon_base = -74.0060 + (i * 0.0001)  # Moving east
        
        mock_data.append({
            "timestamp": timestamp,
            "speed_ms": round(speed, 2),
            "voltage_v": round(voltage, 2),
            "current_a": round(current, 2),
            "power_w": round(power, 2),
            "energy_j": round(cumulative_energy, 2),
            "distance_m": round(cumulative_distance, 2),
            "latitude": round(lat_base + random.gauss(0, 0.0001), 6),
            "longitude": round(lon_base + random.gauss(0, 0.0001), 6),
        })

    # Reverse to get chronological order
    return pd.DataFrame(mock_data[::-1])


# --- Sidebar Configuration (Optimized) ---
def render_sidebar():
    """Render sidebar configuration - isolated to prevent full reruns"""
    with st.sidebar:
        st.header("🔧 Configuration")
        st.subheader("Connection Status")

        # Check connection status
        if st.session_state.connection_error:
            st.error(f"❌ Connection Error: {st.session_state.connection_error}")
            
            st.subheader("📊 Data Source")
            use_mock = st.checkbox(
                "Use Mock Data for Demo",
                value=st.session_state.use_mock_data,
                help="Generate simulated telemetry data for testing",
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


# --- Main Dashboard Logic ---
def main():
    st.markdown(
        f'<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard V5</h1>',
        unsafe_allow_html=True,
    )

    # Initialize connection status only once
    if st.session_state.connection_error is None and st.session_state.ably_client is None:
        client_status, error_msg = get_ably_connection_status()
        if client_status != "available":
            st.session_state.connection_error = error_msg

    # Render sidebar
    render_sidebar()

    # Get current data
    if st.session_state.connection_error and not st.session_state.use_mock_data:
        display_connection_issues()
        current_display_df = pd.DataFrame(columns=PLACEHOLDER_COLS)
    elif st.session_state.use_mock_data:
        current_display_df = generate_mock_data()
        st.info("🎭 Currently displaying simulated telemetry data for demonstration purposes")
    else:
        if st.session_state.telemetry_data_deque:
            current_display_df = pd.DataFrame(list(st.session_state.telemetry_data_deque))
            current_display_df = current_display_df.reindex(columns=PLACEHOLDER_COLS)
            if "timestamp" in current_display_df.columns:
                current_display_df["timestamp"] = pd.to_datetime(current_display_df["timestamp"])
        else:
            current_display_df = pd.DataFrame(columns=PLACEHOLDER_COLS)
            st.warning(f"⏳ Waiting for real-time data from channel '{TELEMETRY_CHANNEL_NAME}'")

    # Convert DataFrame to hashable format for caching
    df_hash = current_display_df.to_dict('records') if not current_display_df.empty else []

    # --- KPIs (Cached) ---
    kpis = calculate_kpis(df_hash)
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

    # --- Charts (Cached) ---
    st.subheader("📈 Telemetry Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"]
    )

    with tab1:
        st.plotly_chart(create_speed_chart(df_hash), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_power_chart(df_hash), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_efficiency_chart(df_hash), use_container_width=True)
    
    with tab4:
        st.plotly_chart(create_gps_map(df_hash), use_container_width=True)
    
    with tab5:
        st.subheader(f"Raw Telemetry Data (Last {MAX_DATAPOINTS_IN_DASHBOARD} points)")
        if not current_display_df.empty:
            st.dataframe(current_display_df, use_container_width=True)
            
            csv = current_display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available to display")

    # Auto-refresh for mock data (less frequent to reduce load)
    if st.session_state.use_mock_data:
        time.sleep(5)  # Increased interval for efficiency
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'><p>Shell Eco-marathon Telemetry Dashboard V5 | Optimized Performance</p></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
