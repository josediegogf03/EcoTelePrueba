import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import json
import time
import sys
from typing import Dict, Any, List, Optional
import threading
import queue
import asyncio
import uuid
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Ably imports
try:
    from ably import AblyRealtime, AblyRest
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("❌ Ably library not available. Please install: pip install ably")
    st.stop()

# Setup logging
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger("TelemetrySubscriber")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

setup_terminal_logging()

# Configuration
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATAPOINTS = 50000
HISTORICAL_REFRESH_INTERVAL = 30  # seconds
REDUCED_QUEUE_SIZE = 50  # Reduced queue size since we're getting historical data

# Page configuration
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Shell Eco-marathon Real-time Telemetry Dashboard",
    },
)

# CSS styling (keeping the existing styles)
st.markdown(
    """
<style>
    /* Theme-aware color variables */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --error-color: #d62728;
        --text-primary: #262730;
        --text-secondary: #6c757d;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --border-color: #dee2e6;
    }

    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #fafafa;
            --text-secondary: #a0a0a0;
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --border-color: #4a4a4a;
        }
    }

    .main-header {
        font-size: 2.2rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .status-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }

    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
    }

    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.2);
    }

    .status-connecting {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 2px solid #ffc107;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.2);
    }

    .historical-data-panel {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }

    .supabase-panel {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #9c27b0;
    }
</style>
""",
    unsafe_allow_html=True,
)

class HistoricalDataRetriever:
    """Handles retrieval of historical data from Ably"""
    
    def __init__(self, api_key: str, channel_name: str):
        self.api_key = api_key
        self.channel_name = channel_name
        self.rest_client = None
        self.logger = logging.getLogger("HistoricalDataRetriever")
        
    def connect(self):
        """Connect to Ably REST API"""
        try:
            self.rest_client = AblyRest(self.api_key)
            self.logger.info("✅ Connected to Ably REST API")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Ably REST API: {e}")
            return False
    
    def retrieve_historical_data(self, hours_back: int = 24, limit: int = 1000) -> List[Dict]:
        """Retrieve historical messages from Ably"""
        if not self.rest_client:
            self.logger.error("❌ REST client not connected")
            return []
            
        try:
            # Calculate start time (hours back from now)
            start_time = datetime.now() - timedelta(hours=hours_back)
            start_timestamp = int(start_time.timestamp() * 1000)  # Convert to milliseconds
            
            # Get channel
            channel = self.rest_client.channels.get(self.channel_name)
            
            # Retrieve history
            self.logger.info(f"🔍 Retrieving historical data from {hours_back} hours ago...")
            history = channel.history(
                start=start_timestamp,
                limit=limit,
                direction='forwards'
            )
            
            messages = []
            for message in history.items:
                if message.data and isinstance(message.data, dict):
                    # Ensure timestamp is properly formatted
                    msg_data = message.data.copy()
                    if 'timestamp' not in msg_data:
                        msg_data['timestamp'] = datetime.fromtimestamp(message.timestamp/1000).isoformat()
                    messages.append(msg_data)
            
            self.logger.info(f"📊 Retrieved {len(messages)} historical messages")
            return messages
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving historical data: {e}")
            return []
    
    def retrieve_last_message(self) -> Optional[Dict]:
        """Retrieve the last message from the channel"""
        if not self.rest_client:
            return None
            
        try:
            channel = self.rest_client.channels.get(self.channel_name)
            history = channel.history(limit=1)
            
            if history.items:
                message = history.items[0]
                if message.data and isinstance(message.data, dict):
                    msg_data = message.data.copy()
                    if 'timestamp' not in msg_data:
                        msg_data['timestamp'] = datetime.fromtimestamp(message.timestamp/1000).isoformat()
                    return msg_data
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving last message: {e}")
            return None

class TelemetrySubscriber:
    """Enhanced subscriber with historical data integration"""
    
    def __init__(self):
        self.ably_client = None
        self.channel = None
        self.is_connected = False
        self.message_queue = queue.Queue(maxsize=REDUCED_QUEUE_SIZE)  # Reduced queue size
        self.connection_thread = None
        self.stats = {
            "messages_received": 0,
            "last_message_time": None,
            "connection_attempts": 0,
            "errors": 0,
            "last_error": None,
            "historical_messages_loaded": 0,
            "last_historical_refresh": None,
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self.logger = logging.getLogger("TelemetrySubscriber")
        
        # Historical data retriever
        self.historical_retriever = HistoricalDataRetriever(ABLY_API_KEY, CHANNEL_NAME)
        
    def connect(self) -> bool:
        """Connect to Ably and start receiving messages"""
        try:
            with self._lock:
                self.stats["connection_attempts"] += 1

            self.logger.info("🔌 Starting connection to Ably...")
            
            if self._should_run:
                self.disconnect()

            self._stop_event.clear()
            self._should_run = True

            self.connection_thread = threading.Thread(
                target=self._connection_worker, daemon=True
            )
            self.connection_thread.start()
            
            # Also connect historical retriever
            self.historical_retriever.connect()
            
            time.sleep(3)
            return self.is_connected

        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False
            return False
    
    def _connection_worker(self):
        """Worker thread to handle Ably connection"""
        try:
            self.logger.info("🔌 Connection worker starting...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_connection_handler())
        except Exception as e:
            self.logger.error(f"💥 Connection worker error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False
        finally:
            self.logger.info("🛑 Connection worker ended")

    async def _async_connection_handler(self):
        """Handle Ably connection asynchronously"""
        try:
            self.logger.info("🔑 Creating Ably client...")
            self.ably_client = AblyRealtime(ABLY_API_KEY)

            def on_connected(state_change):
                self.logger.info(f"✅ Connected to Ably: {state_change}")
                self.is_connected = True

            def on_disconnected(state_change):
                self.logger.warning(f"❌ Disconnected from Ably: {state_change}")
                self.is_connected = False

            def on_failed(state_change):
                self.logger.error(f"💥 Connection failed: {state_change}")
                self.is_connected = False
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"Connection failed: {state_change}"

            self.ably_client.connection.on("connected", on_connected)
            self.ably_client.connection.on("disconnected", on_disconnected)
            self.ably_client.connection.on("failed", on_failed)
            self.ably_client.connection.on("suspended", on_disconnected)

            self.logger.info("⏳ Waiting for connection...")
            await self.ably_client.connection.once_async("connected")

            self.logger.info(f"📡 Getting channel: {CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(CHANNEL_NAME)

            self.logger.info("📨 Subscribing to messages...")
            await self.channel.subscribe("telemetry_update", self._on_message_received)

            self.logger.info("✅ Successfully subscribed to messages!")

            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)

                if hasattr(self.ably_client.connection, "state"):
                    state = self.ably_client.connection.state
                    if state not in ["connected"]:
                        self.logger.warning(f"⚠️ Connection state: {state}")
                        if state in ["failed", "suspended", "disconnected"]:
                            self.is_connected = False
                            break

            self.logger.info("🔚 Connection loop ended")

        except Exception as e:
            self.logger.error(f"💥 Async connection error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False

    def _on_message_received(self, message):
        """Handle incoming messages from Ably"""
        try:
            self.logger.debug(f"📨 Message received: {message.name}")
            data = message.data

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON decode error: {e}")
                    with self._lock:
                        self.stats["errors"] += 1
                        self.stats["last_error"] = f"JSON decode error: {e}"
                    return

            if not isinstance(data, dict):
                self.logger.warning(f"⚠️ Invalid data type: {type(data)}")
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"Invalid data type: {type(data)}"
                return

            self.logger.debug(f"📊 Data keys: {list(data.keys())}")

            with self._lock:
                # Clear queue if it's full to prevent blocking
                if self.message_queue.full():
                    try:
                        while self.message_queue.qsize() > REDUCED_QUEUE_SIZE // 2:
                            self.message_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.message_queue.put(data)
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = datetime.now()

                self.logger.debug(f"✅ Message queued. Total: {self.stats['messages_received']}")

        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Message error: {e}"

    def load_historical_data(self, hours_back: int = 24) -> List[Dict]:
        """Load historical data from Ably"""
        try:
            self.logger.info(f"📚 Loading historical data from {hours_back} hours ago...")
            historical_messages = self.historical_retriever.retrieve_historical_data(hours_back)
            
            with self._lock:
                self.stats["historical_messages_loaded"] = len(historical_messages)
                self.stats["last_historical_refresh"] = datetime.now()
            
            self.logger.info(f"✅ Loaded {len(historical_messages)} historical messages")
            return historical_messages
            
        except Exception as e:
            self.logger.error(f"❌ Error loading historical data: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Historical data error: {e}"
            return []

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all queued messages"""
        messages = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break

        if messages:
            self.logger.debug(f"📤 Returning {len(messages)} messages")

        return messages

    def disconnect(self):
        """Disconnect from Ably"""
        try:
            self.logger.info("🛑 Disconnecting...")
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False

            if self.ably_client:
                try:
                    self.ably_client.close()
                    self.logger.info("✅ Ably connection closed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably: {e}")

            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=5)
                if self.connection_thread.is_alive():
                    self.logger.warning("⚠️ Connection thread did not stop gracefully")

            self.logger.info("🔚 Disconnection complete")

        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Disconnect error: {e}"
        finally:
            self.ably_client = None
            self.channel = None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self._lock:
            return self.stats.copy()

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "subscriber": None,
        "telemetry_data": pd.DataFrame(),
        "historical_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "last_historical_refresh": None,
        "auto_refresh": True,
        "auto_historical_refresh": True,
        "dynamic_charts": [],
        "active_tab": 0,
        "is_auto_refresh": False,
        "scroll_position": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def merge_historical_and_realtime_data(historical_df: pd.DataFrame, realtime_df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical and realtime data, removing duplicates"""
    if historical_df.empty and realtime_df.empty:
        return pd.DataFrame()
    
    if historical_df.empty:
        return realtime_df.copy()
    
    if realtime_df.empty:
        return historical_df.copy()
    
    # Combine dataframes
    combined_df = pd.concat([historical_df, realtime_df], ignore_index=True)
    
    # Remove duplicates based on timestamp and message_id if available
    if 'timestamp' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
        
        # Remove duplicates based on timestamp and message_id
        if 'message_id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'message_id'], keep='last')
        else:
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Limit to MAX_DATAPOINTS
    if len(combined_df) > MAX_DATAPOINTS:
        combined_df = combined_df.tail(MAX_DATAPOINTS)
    
    return combined_df.reset_index(drop=True)

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate KPIs with enhanced error handling"""
    default_kpis = {
        "total_energy_mj": 0.0,
        "max_speed_ms": 0.0,
        "avg_speed_ms": 0.0,
        "total_distance_km": 0.0,
        "avg_power_w": 0.0,
        "efficiency_km_per_mj": 0.0,
        "max_acceleration": 0.0,
        "avg_gyro_magnitude": 0.0,
    }

    if df.empty:
        return default_kpis

    try:
        numeric_cols = [
            "energy_j", "speed_ms", "distance_m", "power_w",
            "total_acceleration", "gyro_x", "gyro_y", "gyro_z",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        kpis = default_kpis.copy()

        if "energy_j" in df.columns and len(df) > 0:
            kpis["total_energy_mj"] = max(0, df["energy_j"].iloc[-1] / 1_000_000)

        if "speed_ms" in df.columns:
            speed_data = df["speed_ms"].dropna()
            if not speed_data.empty:
                kpis["max_speed_ms"] = max(0, speed_data.max())
                kpis["avg_speed_ms"] = max(0, speed_data.mean())

        if "distance_m" in df.columns and len(df) > 0:
            kpis["total_distance_km"] = max(0, df["distance_m"].iloc[-1] / 1000)

        if "power_w" in df.columns:
            power_data = df["power_w"].dropna()
            if not power_data.empty:
                kpis["avg_power_w"] = max(0, power_data.mean())

        if kpis["total_energy_mj"] > 0:
            kpis["efficiency_km_per_mj"] = kpis["total_distance_km"] / kpis["total_energy_mj"]

        if "total_acceleration" in df.columns:
            accel_data = df["total_acceleration"].dropna()
            if not accel_data.empty:
                kpis["max_acceleration"] = max(0, accel_data.max())

        if all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z"]):
            gyro_data = df[["gyro_x", "gyro_y", "gyro_z"]].dropna()
            if not gyro_data.empty:
                gyro_magnitude = np.sqrt(
                    gyro_data["gyro_x"] ** 2 + gyro_data["gyro_y"] ** 2 + gyro_data["gyro_z"] ** 2
                )
                kpis["avg_gyro_magnitude"] = max(0, gyro_magnitude.mean())

        return kpis

    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis

def render_kpi_header(kpis: Dict[str, float]):
    """Render KPI header"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📏 Distance", f"{kpis['total_distance_km']:.2f} km")
        st.metric("🔋 Energy", f"{kpis['total_energy_mj']:.2f} MJ")

    with col2:
        st.metric("🚀 Max Speed", f"{kpis['max_speed_ms']:.1f} m/s")
        st.metric("💡 Avg Power", f"{kpis['avg_power_w']:.1f} W")

    with col3:
        st.metric("🏃 Avg Speed", f"{kpis['avg_speed_ms']:.1f} m/s")
        st.metric("♻️ Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ")

    with col4:
        st.metric("📈 Max Acc.", f"{kpis['max_acceleration']:.2f} m/s²")
        st.metric("🎯 Avg Gyro", f"{kpis['avg_gyro_magnitude']:.2f} °/s")

def render_connection_status(subscriber, stats):
    """Render connection status in sidebar"""
    if subscriber and subscriber.is_connected:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Disconnected</div>',
            unsafe_allow_html=True,
        )

    # Connection statistics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📨 Messages", stats["messages_received"])
        st.metric("🔌 Attempts", stats["connection_attempts"])
    with col2:
        st.metric("❌ Errors", stats["errors"])
        if stats["last_message_time"]:
            time_since = (datetime.now() - stats["last_message_time"]).total_seconds()
            st.metric("⏱️ Last Msg", f"{time_since:.0f}s ago")
        else:
            st.metric("⏱️ Last Msg", "Never")

def render_historical_data_panel():
    """Render historical data controls"""
    st.sidebar.markdown(
        """
        <div class="historical-data-panel">
            <h3>📚 Historical Data</h3>
            <p>Retrieve past 24 hours of telemetry data from Ably's persistent storage</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Load History", use_container_width=True, help="Load historical data from Ably"):
            if st.session_state.subscriber:
                with st.spinner("Loading historical data..."):
                    historical_messages = st.session_state.subscriber.load_historical_data(24)
                    if historical_messages:
                        historical_df = pd.DataFrame(historical_messages)
                        if "timestamp" in historical_df.columns:
                            historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"])
                        st.session_state.historical_data = historical_df
                        st.success(f"✅ Loaded {len(historical_messages)} historical messages")
                    else:
                        st.warning("⚠️ No historical data available")
            else:
                st.error("❌ Please connect first")
    
    with col2:
        auto_historical = st.checkbox(
            "Auto Load",
            value=st.session_state.get("auto_historical_refresh", True),
            help="Automatically refresh historical data every 30 seconds"
        )
        st.session_state.auto_historical_refresh = auto_historical

    # Display historical data stats
    if not st.session_state.historical_data.empty:
        st.sidebar.success(f"📊 {len(st.session_state.historical_data)} historical records loaded")
        
        if st.session_state.subscriber:
            stats = st.session_state.subscriber.get_stats()
            if stats.get("last_historical_refresh"):
                time_since = (datetime.now() - stats["last_historical_refresh"]).total_seconds()
                st.sidebar.info(f"⏱️ Last refresh: {time_since:.0f}s ago")

def render_supabase_panel():
    """Render Supabase integration panel"""
    st.sidebar.markdown(
        """
        <div class="supabase-panel">
            <h3>🗃️ Past Sessions</h3>
            <p>Retrieve data from previous racing sessions stored in Supabase</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Session selection
    session_options = [
        "Select a session...",
        "Session 2025-01-01 (Practice)",
        "Session 2025-01-02 (Qualification)",
        "Session 2025-01-03 (Race 1)",
        "Session 2025-01-04 (Race 2)",
    ]
    
    selected_session = st.sidebar.selectbox(
        "📅 Select Session",
        options=session_options,
        help="Choose a past session to load data from"
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📥 Load Session", use_container_width=True, disabled=selected_session == "Select a session..."):
            if selected_session != "Select a session...":
                with st.spinner(f"Loading {selected_session}..."):
                    # Placeholder for Supabase integration
                    time.sleep(2)
                    st.info("🚧 Supabase integration coming soon!")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.historical_data = pd.DataFrame()
            st.success("✅ Historical data cleared")

# Chart creation functions (keeping existing ones for brevity)
def create_speed_chart(df: pd.DataFrame):
    """Generate speed chart"""
    if df.empty or "speed_ms" not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.line(df, x="timestamp", y="speed_ms", title="🚗 Vehicle Speed Over Time",
                  labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
                  color_discrete_sequence=["#1f77b4"])
    return fig

def create_power_chart(df: pd.DataFrame):
    """Generate power chart"""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
                        vertical_spacing=0.15)
    
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["voltage_v"], name="Voltage (V)",
                            line=dict(color="#2ca02c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["current_a"], name="Current (A)",
                            line=dict(color="#d62728", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["power_w"], name="Power (W)",
                            line=dict(color="#ff7f0e", width=2)), row=2, col=1)
    
    fig.update_layout(height=500, title_text="⚡ Electrical System Performance")
    return fig

def create_imu_chart(df: pd.DataFrame):
    """Generate IMU chart"""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
                        vertical_spacing=0.25)
    
    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                                line=dict(color=colors_gyro[i], width=2)), row=1, col=1)
    
    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                                line=dict(color=colors_accel[i], width=2)), row=2, col=1)
    
    fig.update_layout(height=600, title_text="🎮 IMU Sensor Data Analysis")
    return fig

def create_efficiency_chart(df: pd.DataFrame):
    """Generate efficiency chart"""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.scatter(df, x="speed_ms", y="power_w", color="voltage_v" if "voltage_v" in df.columns else None,
                     title="⚡ Efficiency Analysis: Speed vs Power Consumption",
                     labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
                     color_continuous_scale="viridis")
    return fig

def create_gps_map(df: pd.DataFrame):
    """Generate GPS map"""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    center_point = dict(lat=df_valid["latitude"].mean(), lon=df_valid["longitude"].mean())
    
    fig = px.scatter_map(df_valid, lat="latitude", lon="longitude",
                         color="speed_ms" if "speed_ms" in df_valid.columns else None,
                         size="power_w" if "power_w" in df_valid.columns else None,
                         hover_data=["speed_ms", "power_w", "voltage_v"] if all(col in df_valid.columns for col in ["speed_ms", "power_w", "voltage_v"]) else None,
                         map_style="open-street-map", title="🛰️ Vehicle Track and Performance",
                         height=400, zoom=15, center=center_point, color_continuous_scale="plasma")
    
    return fig

def render_overview_tab(kpis: Dict[str, float]):
    """Render overview tab"""
    st.markdown("### 📊 Performance Overview")
    st.markdown("Real-time key performance indicators for your Shell Eco-marathon vehicle")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🛣️ Total Distance", f"{kpis['total_distance_km']:.2f} km", help="Distance traveled during the session")
        st.metric("🔋 Energy Consumed", f"{kpis['total_energy_mj']:.2f} MJ", help="Total energy consumption")
    
    with col2:
        st.metric("🚀 Maximum Speed", f"{kpis['max_speed_ms']:.1f} m/s", help="Highest speed achieved")
        st.metric("💡 Average Power", f"{kpis['avg_power_w']:.1f} W", help="Mean power consumption")
    
    with col3:
        st.metric("🏃 Average Speed", f"{kpis['avg_speed_ms']:.1f} m/s", help="Mean speed throughout the session")
        st.metric("♻️ Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ", help="Energy efficiency ratio")
    
    with col4:
        st.metric("📈 Max Acceleration", f"{kpis['max_acceleration']:.2f} m/s²", help="Peak acceleration recorded")
        st.metric("🎯 Avg Gyro Magnitude", f"{kpis['avg_gyro_magnitude']:.2f} °/s", help="Average rotational movement")

def create_optimized_chart(df: pd.DataFrame, chart_func, title: str):
    """Create optimized chart with consistent styling"""
    try:
        fig = chart_func(df)
        if fig:
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                title=dict(font=dict(size=16, color="#1f77b4")),
                margin=dict(l=40, r=40, t=60, b=40),
                height=400,
            )
            return fig
    except Exception as e:
        st.error(f"Error creating {title}: {e}")
        return None

def main():
    """Main dashboard function"""
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Connection Control")
        
        # Connection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    time.sleep(2)
                
                with st.spinner("Connecting..."):
                    st.session_state.subscriber = TelemetrySubscriber()
                    if st.session_state.subscriber.connect():
                        st.success("✅ Connected!")
                        # Load initial historical data
                        historical_messages = st.session_state.subscriber.load_historical_data(24)
                        if historical_messages:
                            historical_df = pd.DataFrame(historical_messages)
                            if "timestamp" in historical_df.columns:
                                historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"])
                            st.session_state.historical_data = historical_df
                    else:
                        st.error("❌ Failed!")
                
                st.session_state.is_auto_refresh = False
                st.rerun()
        
        with col2:
            if st.button("🛑 Disconnect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    st.session_state.subscriber = None
                st.info("🛑 Disconnected")
                st.session_state.is_auto_refresh = False
                st.rerun()
        
        # Connection status
        stats = (
            st.session_state.subscriber.get_stats()
            if st.session_state.subscriber
            else {
                "messages_received": 0,
                "connection_attempts": 0,
                "errors": 0,
                "last_message_time": None,
                "last_error": None,
                "historical_messages_loaded": 0,
                "last_historical_refresh": None,
            }
        )
        
        render_connection_status(st.session_state.subscriber, stats)
        
        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")
        
        st.divider()
        
        # Historical data controls
        render_historical_data_panel()
        
        st.divider()
        
        # Supabase integration
        render_supabase_panel()
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        new_auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        
        if new_auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = new_auto_refresh
            st.session_state.is_auto_refresh = False
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)
        
        st.info(f"📡 Channel: {CHANNEL_NAME}")
    
    # Data processing
    new_messages_count = 0
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        new_messages = st.session_state.subscriber.get_messages()
        
        if new_messages:
            new_messages_count = len(new_messages)
            new_df = pd.DataFrame(new_messages)
            
            if "timestamp" in new_df.columns:
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
            
            if st.session_state.telemetry_data.empty:
                st.session_state.telemetry_data = new_df
            else:
                st.session_state.telemetry_data = pd.concat([st.session_state.telemetry_data, new_df], ignore_index=True)
            
            # Limit telemetry data size
            if len(st.session_state.telemetry_data) > MAX_DATAPOINTS:
                st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_DATAPOINTS)
            
            st.session_state.last_update = datetime.now()
    
    # Auto-refresh historical data
    if (st.session_state.auto_historical_refresh and 
        st.session_state.subscriber and 
        st.session_state.subscriber.is_connected):
        
        # Check if it's time to refresh historical data
        should_refresh = False
        if st.session_state.last_historical_refresh is None:
            should_refresh = True
        else:
            time_since_refresh = (datetime.now() - st.session_state.last_historical_refresh).total_seconds()
            if time_since_refresh > HISTORICAL_REFRESH_INTERVAL:
                should_refresh = True
        
        if should_refresh:
            historical_messages = st.session_state.subscriber.load_historical_data(24)
            if historical_messages:
                historical_df = pd.DataFrame(historical_messages)
                if "timestamp" in historical_df.columns:
                    historical_df["timestamp"] = pd.to_datetime(historical_df["timestamp"])
                st.session_state.historical_data = historical_df
                st.session_state.last_historical_refresh = datetime.now()
    
    # Merge historical and realtime data
    df = merge_historical_and_realtime_data(st.session_state.historical_data, st.session_state.telemetry_data)
    
    # Display empty state
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Getting Started:**\n"
                "1. Ensure maindata.py is running\n"
                "2. Click 'Connect' to start receiving data\n"
                "3. Historical data will be loaded automatically"
            )
        
        with col2:
            with st.expander("🔍 Debug Information"):
                st.json({
                    "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                    "Realtime Messages": stats["messages_received"],
                    "Historical Messages": stats.get("historical_messages_loaded", 0),
                    "Errors": stats["errors"],
                    "Channel": CHANNEL_NAME,
                })
        return
    
    # Data status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"📊 **{len(df):,}** total data points")
    with col2:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col3:
        if not st.session_state.historical_data.empty:
            st.info(f"📚 **{len(st.session_state.historical_data):,}** historical")
    with col4:
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count} new")
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Dashboard tabs
    st.subheader("📈 Dashboard")
    
    tab_names = ["📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", "⚡ Efficiency", "🛰️ GPS", "📃 Data"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:
        render_overview_tab(kpis)
    
    with tabs[1]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_speed_chart, "Speed Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_power_chart, "Power Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart, "IMU Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_efficiency_chart, "Efficiency Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[6]:
        render_kpi_header(kpis)
        
        st.subheader("📃 Raw Telemetry Data")
        
        # Data source breakdown
        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                realtime_count = len(st.session_state.telemetry_data)
                st.metric("🔴 Realtime Messages", realtime_count)
            with col2:
                historical_count = len(st.session_state.historical_data)
                st.metric("📚 Historical Messages", historical_count)
        
        st.warning("ℹ️ Only the **last 100 datapoints** are displayed below. Download the CSV for the complete dataset.")
        st.dataframe(df.tail(100), use_container_width=True, height=400)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    # Auto-refresh functionality
    if (st.session_state.auto_refresh and 
        st.session_state.subscriber and 
        st.session_state.subscriber.is_connected):
        
        if not hasattr(st.session_state, "fragment_rerun") or not st.session_state.fragment_rerun:
            time.sleep(refresh_interval)
            st.session_state.is_auto_refresh = True
            st.rerun()
    
    st.session_state.is_auto_refresh = False
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>"
        "<p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | Enhanced with Historical Data Persistence</p>"
        "<p>🚗 Optimized for performance monitoring and energy efficiency analysis</p>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
