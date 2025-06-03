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

# --- Custom CSS with Updated Colors ---
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
        background: linear-gradient(135deg, #ff6b35, #f7931e);
        border: 2px solid #e55100;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .error-card h3 {
        color: white;
        margin-top: 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff5722, #ff9800);
        border: 2px solid #d84315;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .warning-card h3, .warning-card h4 {
        color: white;
        margin-top: 0;
    }
    .info-card {
        background: linear-gradient(135deg, #2196f3, #03a9f4);
        border: 2px solid #1976d2;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-card h4, .info-card h5 {
        color: white;
        margin-top: 0;
    }
    .info-card ul, .info-card li {
        color: white;
    }
    .info-card code {
        background-color: rgba(255,255,255,0.2);
        color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Cached Functions for Efficiency ---
@st.cache_data(ttl=60)  # Cache for 60 seconds
def calculate_kpis(df_dict):
    """Calculate KPIs from DataFrame dictionary (for caching)"""
    df = pd.DataFrame(df_dict)
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


@st.cache_data(ttl=30)  # Cache for 30 seconds
def create_speed_chart(df_dict):
    """Create speed chart from DataFrame dictionary"""
    df = pd.DataFrame(df_dict)
    if df.empty or "timestamp" not in df.columns or "speed_ms" not in df.columns:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No data)", height=400
        )
    
    df["speed_ms"] = pd.to_numeric(df["speed_ms"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_clean = df.dropna(subset=["timestamp", "speed_ms"]).sort_values("timestamp")
    
    if df_clean.empty:
        return go.Figure().update_layout(
            title="Vehicle Speed Over Time (No valid data)", height=400
        )
    
    fig = px.line(
        df_clean,
        x="timestamp",
        y="speed_ms",
        title="Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_traces(line=dict(color="#4CAF50", width=2))
    return fig


@st.cache_data(ttl=30)
def create_power_chart(df_dict):
    """Create power chart from DataFrame dictionary"""
    df = pd.DataFrame(df_dict)
    if df.empty or not all(
        col in df.columns for col in ["timestamp", "voltage_v", "current_a"]
    ):
        return go.Figure().update_layout(
            title="Electrical Parameters (No data)", height=400
        )
    
    df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
    df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_valid = df.dropna(subset=["timestamp", "voltage_v", "current_a"]).sort_values("timestamp")
    
    if df_valid.empty:
        return go.Figure().update_layout(
            title="Electrical Parameters (No valid data)", height=400
        )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df_valid["timestamp"], 
            y=df_valid["voltage_v"], 
            name="Voltage (V)",
            line=dict(color="#2196F3", width=2)
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_valid["timestamp"], 
            y=df_valid["current_a"], 
            name="Current (A)",
            line=dict(color="#FF9800", width=2)
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Electrical Parameters Over Time",
        height=400,
        xaxis_title="Time",
    )
    fig.update_yaxes(title_text="Voltage (V)", secondary_y=False)
    fig.update_yaxes(title_text="Current (A)", secondary_y=True)
    
    return fig


@st.cache_data(ttl=30)
def create_efficiency_chart(df_dict):
    """Create efficiency chart from DataFrame dictionary"""
    df = pd.DataFrame(df_dict)
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
    
    # Calculate efficiency (avoid division by zero)
    df_copy["efficiency"] = np.where(
        df_copy["energy_j"] > 0,
        (df_copy["distance_m"] / (df_copy["energy_j"] / 1000000)),
        0
    )
    df_copy["efficiency"] = df_copy["efficiency"].replace([np.inf, -np.inf], 0)
    
    # Filter out extreme values for better visualization
    df_copy = df_copy[df_copy["efficiency"] < df_copy["efficiency"].quantile(0.95)]
    
    if df_copy.empty:
        return go.Figure().update_layout(
            title="Efficiency Analysis (No valid data after filtering)", height=400
        )
    
    fig = px.scatter(
        df_copy,
        x="speed_ms",
        y="efficiency",
        color="power_w",
        size="voltage_v",
        title="Efficiency Analysis",
        labels={"speed_ms": "Speed (m/s)", "efficiency": "Efficiency (m/MJ)"},
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=400)
    return fig


@st.cache_data(ttl=30)
def create_gps_map(df_dict):
    """Create GPS map from DataFrame dictionary"""
    df = pd.DataFrame(df_dict)
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
    
    # Filter out unrealistic coordinates
    df_valid = df_valid[
        (df_valid["latitude"].between(-90, 90)) & 
        (df_valid["longitude"].between(-180, 180))
    ]
    
    if df_valid.empty:
        return go.Figure().update_layout(
            title="GPS Track (No valid coordinates)",
            mapbox_style="open-street-map",
            height=400,
        )
    
    fig = px.scatter_mapbox(
        df_valid,
        lat="latitude",
        lon="longitude",
        color="speed_ms",
        size="power_w",
        mapbox_style="open-street-map",
        title="Vehicle Track",
        height=400,
        zoom=12,
        color_continuous_scale="Plasma"
    )
    
    center_lat = df_valid["latitude"].mean()
    center_lon = df_valid["longitude"].mean()
    
    fig.update_layout(
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=12
    )
    
    return fig


# --- Error Handling and Connection Status ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
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
@st.cache_resource
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
@st.cache_data(ttl=2)  # Cache for 2 seconds to simulate real-time updates
def generate_mock_data():
    """Generate realistic mock telemetry data"""
    import random
    
    current_time = datetime.now()
    mock_data = []

    # Generate more realistic data with trends
    base_speed = 15  # Base speed in m/s
    base_voltage = 48  # Base voltage
    base_current = 10  # Base current
    
    for i in range(100):  # More data points for better visualization
        timestamp = current_time - timedelta(seconds=i * 2)
        
        # Add some realistic trends and variations
        time_factor = i / 100.0
        speed_variation = 5 * np.sin(time_factor * 2 * np.pi) + random.gauss(0, 2)
        speed = max(0, base_speed + speed_variation)
        
        voltage = base_voltage + random.gauss(0, 1.5)
        current = max(0, base_current + random.gauss(0, 1.5))
        power = voltage * current
        energy = power * 2  # 2 seconds per data point
        distance = sum([max(0, base_speed + 5 * np.sin(j / 100.0 * 2 * np.pi)) * 2 
                       for j in range(i + 1)])  # Cumulative realistic distance

        # Realistic GPS coordinates (simulating a track)
        lat_base = 40.7128
        lon_base = -74.0060
        track_radius = 0.002
        angle = (i / 100.0) * 4 * np.pi  # Multiple laps
        
        latitude = lat_base + track_radius * np.cos(angle) + random.gauss(0, 0.0001)
        longitude = lon_base + track_radius * np.sin(angle) + random.gauss(0, 0.0001)

        mock_data.append(
            {
                "timestamp": timestamp,
                "speed_ms": round(speed, 2),
                "voltage_v": round(voltage, 2),
                "current_a": round(current, 2),
                "power_w": round(power, 2),
                "energy_j": round(energy, 2),
                "distance_m": round(distance, 2),
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
            }
        )

    return pd.DataFrame(mock_data)


# --- Initialize Session State ---
def initialize_session_state():
    """Initialize session state variables"""
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


# --- Sidebar Component ---
@st.fragment
def render_sidebar():
    """Render sidebar with connection status and controls"""
    st.sidebar.header("🔧 Configuration")
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
            if st.sidebar.button("🔄 Refresh Mock Data"):
                st.cache_data.clear()
        else:
            st.sidebar.warning("⏸️ No data source available")

    else:
        st.sidebar.success("✅ Real-time connection available")
        st.sidebar.info(f"Channel: {TELEMETRY_CHANNEL_NAME}")


# --- Main Dashboard Logic ---
def main():
    # Initialize session state
    initialize_session_state()
    
    st.markdown(
        f'<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard V5</h1>',
        unsafe_allow_html=True,
    )

    # Render sidebar (isolated component)
    render_sidebar()

    # Determine data source
    if st.session_state.connection_error and not st.session_state.use_mock_data:
        # Display connection issues and solutions
        display_connection_issues()
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

    # Convert DataFrame to dict for caching
    df_dict = current_display_df.to_dict('records') if not current_display_df.empty else {}

    # --- KPIs Section (cached) ---
    if df_dict:
        kpis = calculate_kpis(df_dict)
    else:
        kpis = {k: 0 for k in ['total_energy_mj', 'max_speed_ms', 'avg_speed_ms', 'total_distance_km', 'avg_power_w', 'efficiency_km_per_mj']}
    
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

    # --- Charts Section (cached and isolated) ---
    st.subheader("📈 Telemetry Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"]
    )

    with tab1:
        if df_dict:
            st.plotly_chart(create_speed_chart(df_dict), use_container_width=True)
        else:
            st.info("No data available for speed analysis")

    with tab2:
        if df_dict:
            st.plotly_chart(create_power_chart(df_dict), use_container_width=True)
        else:
            st.info("No data available for power analysis")

    with tab3:
        if df_dict:
            st.plotly_chart(create_efficiency_chart(df_dict), use_container_width=True)
        else:
            st.info("No data available for efficiency analysis")

    with tab4:
        if df_dict:
            st.plotly_chart(create_gps_map(df_dict), use_container_width=True)
        else:
            st.info("No data available for GPS tracking")

    with tab5:
        st.subheader(f"Raw Telemetry Data (Last {MAX_DATAPOINTS_IN_DASHBOARD} points)")
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

    # Auto-refresh for mock data (only when using mock data)
    if st.session_state.use_mock_data:
        time.sleep(2)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'><p>Shell Eco-marathon Telemetry Dashboard V5 | Optimized Performance</p></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
