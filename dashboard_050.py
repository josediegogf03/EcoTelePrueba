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

# Disable tracemalloc warnings
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Ably import with error handling
try:
    from ably import AblyRealtime

    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("❌ Ably library not available. Please install: pip install ably")
    st.stop()


# --- UPDATED: Function to set up terminal logging ---
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger("TelemetrySubscriber")

    # Prevent adding handlers multiple times on Streamlit reruns
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


# --- Call the logging setup function once at the start ---
setup_terminal_logging()

# Configuration
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATAPOINTS = 50000

# Page configuration - Optimized for better performance
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

# Optimized CSS - Theme-aware colors and improved layout
# Updated CSS - Remove problematic overview classes
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

    .instructions-container {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 6px solid var(--primary-color);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }

    .instructions-title {
        color: var(--primary-color);
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .instructions-content {
        color: var(--text-primary);
        line-height: 1.7;
        font-size: 1rem;
    }

    .chart-type-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .chart-type-card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid var(--border-color);
        transition: all 0.2s ease;
    }

    .chart-type-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
    }

    .chart-type-name {
        font-weight: 700;
        color: var(--primary-color);
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    .chart-type-desc {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.4;
    }

    /* Sticky header for better navigation */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--bg-primary);
        padding: 1rem 0;
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 1rem;
    }

    /* Improved button styling */
    .stButton > button {
        border-radius: 8px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: transparent;
        color: var(--primary-color);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }

    /* Optimized tab styling */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 50;
        background: var(--bg-primary);
        border-bottom: 2px solid var(--border-color);
        border-radius: 8px 8px 0 0;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        margin: 0 0.25rem;
        transition: all 0.2s ease;
    }

    /* Prevent layout shifts */
    .element-container {
        scroll-margin-top: 120px;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }

        .chart-type-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


class TelemetrySubscriber:
    def __init__(self):
        self.ably_client = None
        self.channel = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.connection_thread = None
        self.stats = {
            "messages_received": 0,
            "last_message_time": None,
            "connection_attempts": 0,
            "errors": 0,
            "last_error": None,
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False

        # Setup logging
        self.logger = logging.getLogger("TelemetrySubscriber")

    def connect(self) -> bool:
        """Connect to Ably and start receiving messages"""
        try:
            with self._lock:
                self.stats["connection_attempts"] += 1

            self.logger.info("🔌 Starting connection to Ably...")

            # Stop any existing connection
            if self._should_run:
                self.disconnect()

            # Clear stop event and start
            self._stop_event.clear()
            self._should_run = True

            # Start connection thread
            self.connection_thread = threading.Thread(
                target=self._connection_worker, daemon=True
            )
            self.connection_thread.start()

            # Wait a bit for connection to establish
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

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the connection coroutine
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

            # Create Ably client
            self.ably_client = AblyRealtime(ABLY_API_KEY)

            # Setup connection event handlers
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
                    self.stats[
                        "last_error"
                    ] = f"Connection failed: {state_change}"

            # Attach connection event handlers
            self.ably_client.connection.on("connected", on_connected)
            self.ably_client.connection.on("disconnected", on_disconnected)
            self.ably_client.connection.on("failed", on_failed)
            self.ably_client.connection.on("suspended", on_disconnected)

            # Wait for connection to establish
            self.logger.info("⏳ Waiting for connection...")
            await self.ably_client.connection.once_async("connected")

            # Get channel and subscribe
            self.logger.info(f"📡 Getting channel: {CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(CHANNEL_NAME)

            # Subscribe to messages
            self.logger.info("📨 Subscribing to messages...")
            await self.channel.subscribe(
                "telemetry_update", self._on_message_received
            )

            self.logger.info("✅ Successfully subscribed to messages!")

            # Keep connection alive
            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)

                # Check connection state
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

            # Extract message data
            data = message.data

            # Parse JSON if it's a string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON decode error: {e}")
                    with self._lock:
                        self.stats["errors"] += 1
                        self.stats["last_error"] = f"JSON decode error: {e}"
                    return

            # Validate data
            if not isinstance(data, dict):
                self.logger.warning(f"⚠️ Invalid data type: {type(data)}")
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = (
                        f"Invalid data type: {type(data)}"
                    )
                return

            self.logger.debug(f"📊 Data keys: {list(data.keys())}")

            # Add to message queue
            with self._lock:
                # Prevent queue from growing too large
                if self.message_queue.qsize() > 100:
                    try:
                        # Remove old messages
                        while self.message_queue.qsize() > 50:
                            self.message_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Add new message
                self.message_queue.put(data)
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = datetime.now()

                self.logger.debug(
                    f"✅ Message queued. Total: {self.stats['messages_received']}"
                )

        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Message error: {e}"

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

            # Stop the connection loop
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False

            # Close Ably connection
            if self.ably_client:
                try:
                    self.ably_client.close()
                    self.logger.info("✅ Ably connection closed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably: {e}")

            # Wait for thread to finish
            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=5)
                if self.connection_thread.is_alive():
                    self.logger.warning(
                        "⚠️ Connection thread did not stop gracefully"
                    )

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
    """Initialize Streamlit session state"""
    defaults = {
        "subscriber": None,
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "active_tab": 0,
        "is_auto_refresh": False,
        "scroll_position": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators with error handling"""
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
        # Ensure numeric columns with safe conversion
        numeric_cols = [
            "energy_j",
            "speed_ms",
            "distance_m",
            "power_w",
            "total_acceleration",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Calculate KPIs with safe operations
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

        # Efficiency calculation with division by zero protection
        if kpis["total_energy_mj"] > 0:
            kpis["efficiency_km_per_mj"] = (
                kpis["total_distance_km"] / kpis["total_energy_mj"]
            )

        if "total_acceleration" in df.columns:
            accel_data = df["total_acceleration"].dropna()
            if not accel_data.empty:
                kpis["max_acceleration"] = max(0, accel_data.max())

        # Gyroscope magnitude calculation
        if all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z"]):
            gyro_data = df[["gyro_x", "gyro_y", "gyro_z"]].dropna()
            if not gyro_data.empty:
                gyro_magnitude = np.sqrt(
                    gyro_data["gyro_x"] ** 2
                    + gyro_data["gyro_y"] ** 2
                    + gyro_data["gyro_z"] ** 2
                )
                kpis["avg_gyro_magnitude"] = max(0, gyro_magnitude.mean())

        return kpis

    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis


def render_kpi_header(kpis: Dict[str, float]):
    """Render a compact performance dashboard at the top of a tab"""
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


def render_overview_tab(kpis: Dict[str, float]):
    """Render the Overview tab with enhanced KPI display using Streamlit native components"""
    st.markdown("### 📊 Performance Overview")
    st.markdown(
        "Real-time key performance indicators for your Shell Eco-marathon vehicle"
    )

    # Create KPI layout using Streamlit columns instead of custom HTML
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🛣️ Total Distance",
            value=f"{kpis['total_distance_km']:.2f} km",
            help="Distance traveled during the session",
        )
        st.metric(
            label="🔋 Energy Consumed",
            value=f"{kpis['total_energy_mj']:.2f} MJ",
            help="Total energy consumption",
        )

    with col2:
        st.metric(
            label="🚀 Maximum Speed",
            value=f"{kpis['max_speed_ms']:.1f} m/s",
            help="Highest speed achieved",
        )
        st.metric(
            label="💡 Average Power",
            value=f"{kpis['avg_power_w']:.1f} W",
            help="Mean power consumption",
        )

    with col3:
        st.metric(
            label="🏃 Average Speed",
            value=f"{kpis['avg_speed_ms']:.1f} m/s",
            help="Mean speed throughout the session",
        )
        st.metric(
            label="♻️ Efficiency",
            value=f"{kpis['efficiency_km_per_mj']:.2f} km/MJ",
            help="Energy efficiency ratio",
        )

    with col4:
        st.metric(
            label="📈 Max Acceleration",
            value=f"{kpis['max_acceleration']:.2f} m/s²",
            help="Peak acceleration recorded",
        )
        st.metric(
            label="🎯 Avg Gyro Magnitude",
            value=f"{kpis['avg_gyro_magnitude']:.2f} °/s",
            help="Average rotational movement",
        )


def render_connection_status(subscriber, stats):
    """Render connection status in sidebar"""
    if subscriber and subscriber.is_connected:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Connected & Receiving Data</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Disconnected</div>',
            unsafe_allow_html=True,
        )

    # Connection stats in compact format
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📨 Messages", stats["messages_received"], delta=None)
        st.metric("🔌 Attempts", stats["connection_attempts"], delta=None)
    with col2:
        st.metric("❌ Errors", stats["errors"], delta=None)
        if stats["last_message_time"]:
            time_since = (
                datetime.now() - stats["last_message_time"]
            ).total_seconds()
            st.metric("⏱️ Last Msg", f"{time_since:.0f}s ago", delta=None)
        else:
            st.metric("⏱️ Last Msg", "Never", delta=None)


def create_optimized_chart(df: pd.DataFrame, chart_func, title: str):
    """Create optimized chart with consistent styling"""
    try:
        fig = chart_func(df)
        if fig:
            # Apply consistent theme-aware styling
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


def create_speed_chart(df: pd.DataFrame):
    """Create speed over time chart"""
    if df.empty or "speed_ms" not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = px.line(
        df,
        x="timestamp",
        y="speed_ms",
        title="🚗 Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4"],
    )
    return fig


def create_power_chart(df: pd.DataFrame):
    """Create power system chart"""
    if df.empty or not all(
        col in df.columns for col in ["voltage_v", "current_a", "power_w"]
    ):
        return go.Figure().add_annotation(
            text="No power data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["voltage_v"],
            name="Voltage (V)",
            line=dict(color="#2ca02c", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["current_a"],
            name="Current (A)",
            line=dict(color="#d62728", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["power_w"],
            name="Power (W)",
            line=dict(color="#ff7f0e", width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=500, title_text="⚡ Electrical System Performance")
    return fig


def create_imu_chart(df: pd.DataFrame):
    """Create IMU (gyroscope and accelerometer) analysis chart"""
    if df.empty or not all(
        col in df.columns
        for col in [
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
        ]
    ):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "🎯 Gyroscope Data (deg/s)",
            "📈 Accelerometer Data (m/s²)",
        ),
        vertical_spacing=0.25,
    )

    # Gyroscope data with distinct colors
    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[axis],
                name=f"Gyro {axis[-1].upper()}",
                line=dict(color=colors_gyro[i], width=2),
            ),
            row=1,
            col=1,
        )

    # Accelerometer data with distinct colors
    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[axis],
                name=f"Accel {axis[-1].upper()}",
                line=dict(color=colors_accel[i], width=2),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=600, title_text="🎮 IMU Sensor Data Analysis")
    return fig


def create_imu_chart_2(df: pd.DataFrame):
    """Create detailed IMU chart with individual subplots"""
    if df.empty or not all(
        col in df.columns
        for col in [
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
        ]
    ):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "🌀 Gyro X",
            "🌀 Gyro Y",
            "🌀 Gyro Z",
            "📊 Accel X",
            "📊 Accel Y",
            "📊 Accel Z",
        ),
        vertical_spacing=0.3,
        horizontal_spacing=0.1,
    )

    # Color schemes
    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]

    # Gyroscope data
    for i, (axis, color) in enumerate(zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[axis],
                name=f"Gyro {axis[-1].upper()}",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )

    # Accelerometer data
    for i, (axis, color) in enumerate(
        zip(["accel_x", "accel_y", "accel_z"], accel_colors)
    ):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[axis],
                name=f"Accel {axis[-1].upper()}",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=2,
            col=i + 1,
        )

    fig.update_layout(height=600, title_text="🎮 Detailed IMU Sensor Analysis")
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Create efficiency analysis chart"""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig = px.scatter(
        df,
        x="speed_ms",
        y="power_w",
        color="voltage_v" if "voltage_v" in df.columns else None,
        title="⚡ Efficiency Analysis: Speed vs Power Consumption",
        labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
        color_continuous_scale="viridis",
    )
    return fig


def create_gps_map(df: pd.DataFrame):
    """Create GPS tracking map"""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    center_point = dict(
        lat=df_valid["latitude"].mean(), lon=df_valid["longitude"].mean()
    )

    fig = px.scatter_map(
        df_valid,
        lat="latitude",
        lon="longitude",
        color="speed_ms" if "speed_ms" in df_valid.columns else None,
        size="power_w" if "power_w" in df_valid.columns else None,
        hover_data=(
            ["speed_ms", "power_w", "voltage_v"]
            if all(
                col in df_valid.columns
                for col in ["speed_ms", "power_w", "voltage_v"]
            )
            else None
        ),
        map_style="open-street-map",
        title="🛰️ Vehicle Track and Performance",
        height=400,
        zoom=12,
        center=center_point,
        color_continuous_scale="plasma",
    )

    return fig


def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns available for plotting"""
    if df.empty:
        return []

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create a dynamic chart based on user configuration with heatmap support"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    if not y_col or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    try:
        if chart_type == "line":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#1f77b4"],
            )
        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#ff7f0e"],
            )
        elif chart_type == "bar":
            recent_df = df.tail(20)
            fig = px.bar(
                recent_df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#2ca02c"],
            )
        elif chart_type == "histogram":
            fig = px.histogram(
                df,
                x=y_col,
                title=f"Distribution of {y_col}",
                color_discrete_sequence=["#d62728"],
            )
        elif chart_type == "heatmap":
            # Create heatmap for correlation analysis
            numeric_cols = get_available_columns(df)
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title=f"🔥 Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                )
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
        else:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#1f77b4"],
            )

        fig.update_layout(height=400)
        return fig

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )


def render_dynamic_charts_section(df: pd.DataFrame):
    """Render the dynamic charts section with proper styling"""

    st.session_state.is_auto_refresh = True

    # Enhanced instructions section with proper HTML rendering
    st.markdown(
        """
    <div class="instructions-container">
        <div class="instructions-title">
            🎯 Create Custom Charts
        </div>
        <div class="instructions-content">
            <p>Click <strong>"Add Chart"</strong> to create custom visualizations with your preferred variables and chart types.</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Chart type information in a grid
    st.markdown(
        """
    <div class="chart-type-grid">
        <div class="chart-type-card">
            <div class="chart-type-name">📈 Line Chart</div>
            <div class="chart-type-desc">Great for time series data and trends</div>
        </div>
        <div class="chart-type-card">
            <div class="chart-type-name">🔵 Scatter Plot</div>
            <div class="chart-type-desc">Perfect for correlation analysis between variables</div>
        </div>
        <div class="chart-type-card">
            <div class="chart-type-name">📊 Bar Chart</div>
            <div class="chart-type-desc">Good for comparing recent values and discrete data</div>
        </div>
        <div class="chart-type-card">
            <div class="chart-type-name">📉 Histogram</div>
            <div class="chart-type-desc">Shows data distribution and frequency patterns</div>
        </div>
        <div class="chart-type-card">
            <div class="chart-type-name">🔥 Heatmap</div>
            <div class="chart-type-desc">Visualizes correlations between all numeric variables</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []

    if not available_columns:
        st.warning(
            "⏳ No numeric data available for creating charts. Connect and wait for data."
        )
        return

    # Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart"):
            try:
                new_chart = {
                    "id": str(uuid.uuid4()),
                    "title": "New Chart",
                    "chart_type": "line",
                    "x_axis": (
                        "timestamp"
                        if "timestamp" in df.columns
                        else available_columns[0]
                    ),
                    "y_axis": available_columns[0] if available_columns else None,
                }
                st.session_state.dynamic_charts.append(new_chart)
                st.session_state.is_auto_refresh = False
                st.rerun()
            except Exception as e:
                st.error(f"Error adding chart: {e}")

    with col2:
        if st.session_state.dynamic_charts:
            st.success(
                f"📈 {len(st.session_state.dynamic_charts)} custom chart(s) active"
            )

    # Display charts
    if st.session_state.dynamic_charts:
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            try:
                with st.container(border=True):
                    # Chart controls in a more compact layout
                    col1, col2, col3, col4, col5 = st.columns(
                        [2, 1.5, 1.5, 1.5, 0.5]
                    )

                    with col1:
                        new_title = st.text_input(
                            "Title",
                            value=chart_config.get("title", "New Chart"),
                            key=f"title_{chart_config['id']}",
                        )
                        if new_title != chart_config.get("title"):
                            st.session_state.dynamic_charts[i]["title"] = new_title

                    with col2:
                        new_type = st.selectbox(
                            "Type",
                            options=[
                                "line",
                                "scatter",
                                "bar",
                                "histogram",
                                "heatmap",
                            ],
                            index=[
                                "line",
                                "scatter",
                                "bar",
                                "histogram",
                                "heatmap",
                            ].index(chart_config.get("chart_type", "line")),
                            key=f"type_{chart_config['id']}",
                        )
                        if new_type != chart_config.get("chart_type"):
                            st.session_state.dynamic_charts[i][
                                "chart_type"
                            ] = new_type

                    with col3:
                        if chart_config.get("chart_type", "line") not in [
                            "histogram",
                            "heatmap",
                        ]:
                            x_options = (
                                ["timestamp"] + available_columns
                                if "timestamp" in df.columns
                                else available_columns
                            )
                            current_x = chart_config.get("x_axis", x_options[0])
                            if current_x not in x_options and x_options:
                                current_x = x_options[0]

                            if x_options:
                                new_x = st.selectbox(
                                    "X-Axis",
                                    options=x_options,
                                    index=(
                                        x_options.index(current_x)
                                        if current_x in x_options
                                        else 0
                                    ),
                                    key=f"x_{chart_config['id']}",
                                )
                                if new_x != chart_config.get("x_axis"):
                                    st.session_state.dynamic_charts[i][
                                        "x_axis"
                                    ] = new_x

                    with col4:
                        if chart_config.get("chart_type", "line") != "heatmap":
                            if available_columns:
                                current_y = chart_config.get(
                                    "y_axis", available_columns[0]
                                )
                                if current_y not in available_columns:
                                    current_y = available_columns[0]

                                new_y = st.selectbox(
                                    "Y-Axis",
                                    options=available_columns,
                                    index=(
                                        available_columns.index(current_y)
                                        if current_y in available_columns
                                        else 0
                                    ),
                                    key=f"y_{chart_config['id']}",
                                )
                                if new_y != chart_config.get("y_axis"):
                                    st.session_state.dynamic_charts[i][
                                        "y_axis"
                                    ] = new_y

                    with col5:
                        if st.button(
                            "🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"
                        ):
                            try:
                                st.session_state.dynamic_charts.pop(i)
                                st.session_state.is_auto_refresh = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")

                    # Display chart
                    try:
                        if chart_config.get("chart_type") == "heatmap" or chart_config.get(
                            "y_axis"
                        ):
                            fig = create_dynamic_chart(df, chart_config)
                            if fig:
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"chart_{chart_config['id']}",
                                )
                        else:
                            st.warning(
                                "Please select a Y-axis variable for this chart."
                            )
                    except Exception as e:
                        st.error(f"Error creating chart: {e}")

            except Exception as e:
                st.error(f"Error rendering chart {i}: {e}")


def main():
    """Optimized main dashboard function"""
    # Sticky header
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    initialize_session_state()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("🔧 Connection Control")

        # Connection buttons in a single row
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
            }
        )

        render_connection_status(st.session_state.subscriber, stats)

        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        new_auto_refresh = st.checkbox(
            "🔄 Auto Refresh", value=st.session_state.auto_refresh
        )

        if new_auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = new_auto_refresh
            st.session_state.is_auto_refresh = False

        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)

        st.info(f"📡 Channel: {CHANNEL_NAME}")

    # ---------- Data ingestion ----------
    new_messages_count = 0
    if (
        st.session_state.subscriber
        and st.session_state.subscriber.is_connected
    ):
        new_messages = st.session_state.subscriber.get_messages()

        if new_messages:
            new_messages_count = len(new_messages)
            new_df = pd.DataFrame(new_messages)

            if "timestamp" in new_df.columns:
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

            if st.session_state.telemetry_data.empty:
                st.session_state.telemetry_data = new_df
            else:
                st.session_state.telemetry_data = pd.concat(
                    [st.session_state.telemetry_data, new_df], ignore_index=True
                )

            if len(st.session_state.telemetry_data) > MAX_DATAPOINTS:
                st.session_state.telemetry_data = (
                    st.session_state.telemetry_data.tail(MAX_DATAPOINTS)
                )

            st.session_state.last_update = datetime.now()

    df = st.session_state.telemetry_data.copy()

    # ---------- Empty state ----------
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")

        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Getting Started:**\n"
                "1. Ensure maindata.py is running\n"
                "2. Click 'Connect' to start receiving data"
            )

        with col2:
            with st.expander("🔍 Debug Information"):
                st.json(
                    {
                        "Connected": st.session_state.subscriber.is_connected
                        if st.session_state.subscriber
                        else False,
                        "Messages": stats["messages_received"],
                        "Errors": stats["errors"],
                        "Channel": CHANNEL_NAME,
                    }
                )
        return  # Stop rendering rest of UI until data arrives

    # ---------- Status row ----------
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points collected")
    with col2:
        st.info(
            f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**"
        )
    with col3:
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")

    # ---------- KPIs ----------
    kpis = calculate_kpis(df)

    # ---------- Tabs ----------
    st.subheader("📈 Dashboard")

    tab_names = [
        "📊 Overview",
        "🚗 Speed",
        "⚡ Power",
        "🎮 IMU",
        "🎮 IMU Detail",
        "⚡ Efficiency",
        "🛰️ GPS",
        "📈 Custom",
        "📃 Data",
    ]
    tabs = st.tabs(tab_names)

    # ------ Overview ------
    with tabs[0]:
        render_overview_tab(kpis)

    # ------ Speed ------
    with tabs[1]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_speed_chart, "Speed Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------ Power ------
    with tabs[2]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_power_chart, "Power Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------ IMU ------
    with tabs[3]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart, "IMU Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------ IMU Detail ------
    with tabs[4]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart_2, "IMU Detail Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------ Efficiency ------
    with tabs[5]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(
            df, create_efficiency_chart, "Efficiency Chart"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------ GPS ------
    with tabs[6]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------ Custom ------
    with tabs[7]:
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)

    # ------ Data ------
    with tabs[8]:
        render_kpi_header(kpis)

        st.subheader("📃 Raw Telemetry Data")
        st.warning(
            "ℹ️ Only the **last 100 datapoints** are displayed below. "
            "Download the CSV for the complete dataset."
        )
        st.dataframe(df.tail(100), use_container_width=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ---------- Auto-refresh ----------
    if (
        st.session_state.auto_refresh
        and st.session_state.subscriber
        and st.session_state.subscriber.is_connected
    ):
        if (
            not hasattr(st.session_state, "fragment_rerun")
            or not st.session_state.fragment_rerun
        ):
            time.sleep(refresh_interval)
            st.session_state.is_auto_refresh = True
            st.rerun()

    st.session_state.is_auto_refresh = False

    # ---------- Footer ----------
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>"
        "<p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | Real-time Data Visualization & Analysis</p>"
        "<p>🚗 Optimized for performance monitoring and energy efficiency analysis</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
