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
import threading
from collections import deque
import json

# Try to import Ably Realtime, fall back if not installed
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    AblyRealtime = None

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
    page_title="Shell Eco-marathon Telemetry Dashboard ",
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
.status-connected {
    color: #4CAF50;
    font-weight: bold;
}
.status-disconnected {
    color: #f44336;
    font-weight: bold;
}
.status-connecting {
    color: #ff9800;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- WebSocket Connection Manager ---
class AblyWebSocketManager:
    def __init__(self):
        self.realtime = None
        self.channel = None
        self.connection_status = "disconnected"
        self.last_error = None
        self.data_deque = deque(maxlen=MAX_DATAPOINTS_IN_DASHBOARD)
        self.lock = threading.Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def _validate_api_key(self) -> bool:
        """Validate Ably API key format and availability"""
        if not ABLY_API_KEY:
            self.last_error = "ABLY_API_KEY is empty or None"
            return False
        
        if len(ABLY_API_KEY) < 20:  # Basic length check
            self.last_error = "ABLY_API_KEY appears to be too short"
            return False
        
        return True
        
    async def connect(self):
        """Connect to Ably with error handling and retry logic"""
        if not ABLY_AVAILABLE:
            self.last_error = "Ably library not available"
            self.connection_status = "error"
            return False
            
        if not self._validate_api_key():
            self.connection_status = "error"
            return False
            
        try:
            self.connection_status = "connecting"
            
            # Create AblyRealtime instance with just the API key (like in maindata.py)
            self.realtime = AblyRealtime(ABLY_API_KEY)
            
            # Set up connection state listeners if supported
            try:
                self.realtime.connection.on('connected', self._on_connected)
                self.realtime.connection.on('disconnected', self._on_disconnected)
                self.realtime.connection.on('suspended', self._on_suspended)
                self.realtime.connection.on('failed', self._on_failed)
            except AttributeError:
                logging.warning("Connection event listeners not supported in this Ably version")
            
            # Wait for connection with timeout
            try:
                await asyncio.wait_for(
                    self.realtime.connection.once_async('connected'), 
                    timeout=15.0
                )
            except (AttributeError, TypeError):
                # Fallback for older versions - just wait a bit
                logging.info("Using fallback connection method...")
                await asyncio.sleep(3)
                
                # Check connection state if available
                if hasattr(self.realtime.connection, 'state'):
                    if self.realtime.connection.state != 'connected':
                        raise Exception(f"Connection failed, state: {self.realtime.connection.state}")
            
            self.channel = self.realtime.channels.get(TELEMETRY_CHANNEL_NAME)
            
            # Subscribe to messages
            await self.channel.subscribe('telemetry_update', self._on_message)
            
            self.connection_status = "connected"
            self.reconnect_attempts = 0
            self.last_error = None
            logging.info("Successfully connected to Ably WebSocket")
            return True
            
        except asyncio.TimeoutError:
            self.last_error = "Connection timeout"
            self.connection_status = "timeout"
            logging.error("Ably connection timeout")
            return False
        except Exception as e:
            self.last_error = str(e)
            self.connection_status = "error"
            logging.error(f"Error connecting to Ably: {e}")
            return False
    
    def _on_connected(self, state_change=None):
        """Handle successful connection"""
        self.connection_status = "connected"
        self.reconnect_attempts = 0
        logging.info("Ably connection established")
    
    def _on_disconnected(self, state_change=None):
        """Handle disconnection"""
        self.connection_status = "disconnected"
        reason = getattr(state_change, 'reason', 'Unknown') if state_change else 'Unknown'
        logging.warning(f"Ably disconnected: {reason}")
    
    def _on_suspended(self, state_change=None):
        """Handle connection suspension"""
        self.connection_status = "suspended"
        reason = getattr(state_change, 'reason', 'Unknown') if state_change else 'Unknown'
        logging.warning(f"Ably connection suspended: {reason}")
    
    def _on_failed(self, state_change=None):
        """Handle connection failure"""
        self.connection_status = "failed"
        reason = getattr(state_change, 'reason', 'Unknown') if state_change else 'Unknown'
        logging.error(f"Ably connection failed: {reason}")
        self._schedule_reconnect()
    
    def _on_message(self, message):
        """Handle incoming telemetry messages"""
        try:
            data = message.data
            if isinstance(data, str):
                data = json.loads(data)
            
            # Validate required fields
            required_fields = ['timestamp', 'speed_ms', 'voltage_v', 'current_a']
            if all(field in data for field in required_fields):
                with self.lock:
                    self.data_deque.append(data)
                logging.debug(f"Received telemetry data: {data.get('timestamp', 'unknown')}")
            else:
                logging.warning(f"Received incomplete telemetry data: {data}")
                
        except Exception as e:
            logging.error(f"Error processing message: {e}")
    
    def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logging.info(f"Scheduling reconnection attempt {self.reconnect_attempts}")
            # In a real application, you'd want to implement exponential backoff
            threading.Timer(5.0, self._attempt_reconnect).start()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect"""
        try:
            asyncio.run(self.connect())
        except Exception as e:
            logging.error(f"Reconnection attempt failed: {e}")
    
    def get_data(self):
        """Get current data safely"""
        with self.lock:
            return list(self.data_deque)
    
    def get_status(self):
        """Get connection status and error info"""
        return {
            'status': self.connection_status,
            'error': self.last_error,
            'data_count': len(self.data_deque)
        }
    
    async def disconnect(self):
        """Gracefully disconnect"""
        if self.realtime:
            try:
                # Try async close first, fall back to sync
                if hasattr(self.realtime, 'close'):
                    if asyncio.iscoroutinefunction(self.realtime.close):
                        await asyncio.wait_for(self.realtime.close(), timeout=5.0)
                    else:
                        self.realtime.close()
                
                self.connection_status = "disconnected"
                logging.info("Ably connection closed")
            except Exception as e:
                logging.error(f"Error closing Ably connection: {e}")

# --- Initialize Session State ---
if "websocket_manager" not in st.session_state:
    st.session_state.websocket_manager = AblyWebSocketManager()

if "connection_initialized" not in st.session_state:
    st.session_state.connection_initialized = False

if "use_mock_data" not in st.session_state:
    st.session_state.use_mock_data = False

if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None

# --- Initialize WebSocket Connection ---
async def initialize_websocket():
    """Initialize WebSocket connection once"""
    if not st.session_state.connection_initialized:
        success = await st.session_state.websocket_manager.connect()
        st.session_state.connection_initialized = True
        return success
    return True

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
            "timestamp": ts.isoformat(),
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
        
        # Get connection status
        status_info = st.session_state.websocket_manager.get_status()
        
        st.subheader("📡 Connection Status")
        
        # Status display with colors
        status = status_info['status']
        if status == 'connected':
            st.markdown('<p class="status-connected">✅ Connected</p>', unsafe_allow_html=True)
            st.success(f"Channel: {TELEMETRY_CHANNEL_NAME}")
            st.info(f"Data points: {status_info['data_count']}")
        elif status == 'connecting':
            st.markdown('<p class="status-connecting">🔄 Connecting...</p>', unsafe_allow_html=True)
        elif status in ['disconnected', 'error', 'timeout', 'failed', 'suspended']:
            st.markdown('<p class="status-disconnected">❌ Disconnected</p>', unsafe_allow_html=True)
            if status_info['error']:
                st.error(f"Error: {status_info['error']}")
        
        # Manual reconnect button
        if status != 'connected':
            if st.button("🔄 Reconnect"):
                asyncio.run(st.session_state.websocket_manager.connect())
                st.rerun()
        
        # Data source selection
        st.subheader("📊 Data Source")
        use_mock = st.checkbox(
            "Use Mock Data for Demo",
            value=st.session_state.use_mock_data,
            help="Switch to simulated data if real connection fails"
        )
        
        if use_mock != st.session_state.use_mock_data:
            st.session_state.use_mock_data = use_mock
            st.rerun()
        
        if use_mock:
            st.info("🎭 Using simulated telemetry data")
        elif status == 'connected':
            st.success("📡 Using real-time WebSocket data")
        else:
            st.warning("⏸️ No data source available")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (3s)", value=True)
        if auto_refresh:
            time.sleep(3)
            st.rerun()
        
        # Display last update time
        if st.session_state.last_update_time:
            st.caption(f"Last updated: {st.session_state.last_update_time.strftime('%H:%M:%S')}")

# --- Main Application ---
def main():
    st.markdown(
        '<h1 class="main-header">'
        '🏎️ Shell Eco-marathon Telemetry Dashboard V7'
        '</h1>',
        unsafe_allow_html=True,
    )

    render_sidebar()

    # Initialize WebSocket connection
    if not st.session_state.connection_initialized:
        with st.spinner("Initializing WebSocket connection..."):
            asyncio.run(initialize_websocket())

    # Update last update time
    st.session_state.last_update_time = datetime.now()

    # --- Build DataFrame ---
    if st.session_state.use_mock_data:
        df = generate_mock_data()
        st.info("🎭 Displaying simulated telemetry data")
    else:
        # Get data from WebSocket manager
        websocket_data = st.session_state.websocket_manager.get_data()
        
        if websocket_data:
            df = pd.DataFrame(websocket_data)
            # Ensure all required columns exist
            for col in PLACEHOLDER_COLS:
                if col not in df.columns:
                    df[col] = 0
            df = df.reindex(columns=PLACEHOLDER_COLS)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            df = pd.DataFrame(columns=PLACEHOLDER_COLS)
            status = st.session_state.websocket_manager.get_status()
            if status['status'] == 'connected':
                st.info(f"⏳ Waiting for data on '{TELEMETRY_CHANNEL_NAME}'")
            else:
                st.warning("🔌 No connection to data source")

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
        "Shell Eco-marathon Telemetry Dashboard | Real-Time WebSocket via Ably"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
