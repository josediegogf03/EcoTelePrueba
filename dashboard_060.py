# dashboard_060.py - Enhanced Dashboard with Supabase Integration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import logging
import json
import time
import sys
from typing import Dict, Any, List, Optional
import threading
import queue
import asyncio
import uuid

# Disables tracemalloc warnings that can appear in Streamlit environments.
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Import required libraries with error checking
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("❌ Ably library not available. Please install: pip install ably")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.error("❌ Supabase library not available. Please install: pip install supabase")

if not ABLY_AVAILABLE or not SUPABASE_AVAILABLE:
    st.stop()

# Function to set up terminal logging
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
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"
MAX_REALTIME_DATAPOINTS = 1000
DB_REFRESH_INTERVAL = 10  # seconds

# Configure Streamlit page
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard v0.6",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Shell Eco-marathon Real-time Telemetry Dashboard with Database Integration",
    },
)

# Enhanced CSS styling
st.markdown(
    """
<style>
    /* Theme-aware color variables */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --error-color: #d62728;
        --info-color: #17a2b8;
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

    .session-card {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .session-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .session-active {
        border-color: var(--success-color);
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }

    .data-source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-realtime {
        background: var(--success-color);
        color: white;
    }

    .badge-historical {
        background: var(--info-color);
        color: white;
    }

    .badge-combined {
        background: var(--primary-color);
        color: white;
    }

    .instructions-container {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 6px solid var(--primary-color);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }

    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--bg-primary);
        padding: 1rem 0;
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 1rem;
    }

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

    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


class SupabaseManager:
    """Manages Supabase database connections and queries"""
    
    def __init__(self):
        self.client = None
        self.logger = logging.getLogger("SupabaseManager")
        self.connect()
    
    def connect(self) -> bool:
        """Connect to Supabase database"""
        try:
            self.client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
            self.logger.info("✅ Connected to Supabase database")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all available sessions from the database"""
        try:
            if not self.client:
                return []
            
            # Get distinct sessions with their metadata
            response = self.client.table(SUPABASE_TABLE_NAME).select(
                "session_id, timestamp"
            ).execute()
            
            if not response.data:
                return []
            
            # Group by session_id and get min/max timestamps
            sessions_dict = {}
            for row in response.data:
                session_id = row['session_id']
                timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                
                if session_id not in sessions_dict:
                    sessions_dict[session_id] = {
                        'session_id': session_id,
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'record_count': 1
                    }
                else:
                    sessions_dict[session_id]['start_time'] = min(
                        sessions_dict[session_id]['start_time'], timestamp
                    )
                    sessions_dict[session_id]['end_time'] = max(
                        sessions_dict[session_id]['end_time'], timestamp
                    )
                    sessions_dict[session_id]['record_count'] += 1
            
            # Convert to list and sort by start time (newest first)
            sessions = list(sessions_dict.values())
            sessions.sort(key=lambda x: x['start_time'], reverse=True)
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting sessions: {e}")
            return []
    
    def get_session_data(self, session_id: str, limit: int = 10000) -> pd.DataFrame:
        """Get telemetry data for a specific session"""
        try:
            if not self.client:
                return pd.DataFrame()
            
            response = self.client.table(SUPABASE_TABLE_NAME).select("*").eq(
                "session_id", session_id
            ).order("timestamp", desc=False).limit(limit).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Ensure timestamp is properly parsed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add calculated fields
            if 'accel_x' in df.columns and 'accel_y' in df.columns and 'accel_z' in df.columns:
                df['total_acceleration'] = np.sqrt(
                    df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2
                )
            
            if 'gyro_x' in df.columns and 'gyro_y' in df.columns and 'gyro_z' in df.columns:
                df['gyro_magnitude'] = np.sqrt(
                    df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting session data: {e}")
            return pd.DataFrame()
    
    def get_recent_data(self, minutes: int = 10) -> pd.DataFrame:
        """Get recent telemetry data from the last N minutes"""
        try:
            if not self.client:
                return pd.DataFrame()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            cutoff_str = cutoff_time.isoformat()
            
            response = self.client.table(SUPABASE_TABLE_NAME).select("*").gte(
                "timestamp", cutoff_str
            ).order("timestamp", desc=False).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Ensure timestamp is properly parsed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting recent data: {e}")
            return pd.DataFrame()


class TelemetrySubscriber:
    """Manages connection to Ably and subscribes to telemetry data."""

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
        self.logger = logging.getLogger("TelemetrySubscriber")

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
                if self.message_queue.qsize() > 100:
                    try:
                        while self.message_queue.qsize() > 50:
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
    """Initialize Streamlit session state variables"""
    defaults = {
        "subscriber": None,
        "supabase_manager": None,
        "telemetry_data": pd.DataFrame(),
        "historical_data": pd.DataFrame(),
        "combined_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "last_db_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "active_tab": 0,
        "selected_session": None,
        "available_sessions": [],
        "data_source": "combined",  # "realtime", "historical", "combined"
        "db_refresh_counter": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators from telemetry data"""
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
            "total_acceleration", "gyro_x", "gyro_y", "gyro_z"
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
                    gyro_data["gyro_x"] ** 2 + 
                    gyro_data["gyro_y"] ** 2 + 
                    gyro_data["gyro_z"] ** 2
                )
                kpis["avg_gyro_magnitude"] = max(0, gyro_magnitude.mean())

        return kpis

    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis


def render_kpi_header(kpis: Dict[str, float], data_source: str = "combined"):
    """Render KPI header with data source indicator"""
    
    # Data source indicators
    source_badges = {
        "realtime": "🔴 Real-time",
        "historical": "📊 Historical", 
        "combined": "🔄 Combined"
    }
    
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown(f'<div class="data-source-badge badge-{data_source}">{source_badges.get(data_source, "Unknown")}</div>', unsafe_allow_html=True)
    
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


def render_session_selector():
    """Render session selection interface"""
    st.subheader("🗂️ Session Management")
    
    # Get available sessions
    sessions = st.session_state.supabase_manager.get_sessions()
    st.session_state.available_sessions = sessions
    
    if not sessions:
        st.info("📭 No sessions found in database. Start the bridge to create a session.")
        return None
    
    # Create session options
    session_options = []
    for session in sessions:
        duration = session['end_time'] - session['start_time']
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        option_text = (
            f"📅 {session['start_time'].strftime('%Y-%m-%d %H:%M:%S')} "
            f"({duration_str}, {session['record_count']} records)"
        )
        session_options.append(option_text)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_index = st.selectbox(
            "Select Session",
            range(len(session_options)),
            format_func=lambda i: session_options[i],
            key="session_selector"
        )
    
    with col2:
        if st.button("🔄 Refresh Sessions", help="Reload session list from database"):
            st.rerun()
    
    if selected_index is not None:
        selected_session = sessions[selected_index]
        
        # Display session details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Session ID:** {selected_session['session_id'][:8]}...")
        with col2:
            st.info(f"**Duration:** {str(selected_session['end_time'] - selected_session['start_time']).split('.')[0]}")
        with col3:
            st.info(f"**Records:** {selected_session['record_count']}")
        
        return selected_session
    
    return None


def render_connection_status(subscriber, supabase_manager, stats):
    """Render connection status for both Ably and Supabase"""
    st.subheader("🔗 Connection Status")
    
    # Ably connection status
    if subscriber and subscriber.is_connected:
        st.markdown(
            '<div class="status-indicator status-connected">🔴 Ably: Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-indicator status-disconnected">🔴 Ably: Disconnected</div>',
            unsafe_allow_html=True,
        )
    
    # Supabase connection status
    if supabase_manager and supabase_manager.client:
        st.markdown(
            '<div class="status-indicator status-connected">💾 Database: Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-indicator status-disconnected">💾 Database: Disconnected</div>',
            unsafe_allow_html=True,
        )
    
    # Statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📨 RT Messages", stats["messages_received"])
        st.metric("🔌 Attempts", stats["connection_attempts"])
    with col2:
        st.metric("❌ Errors", stats["errors"])
        if stats["last_message_time"]:
            time_since = (datetime.now() - stats["last_message_time"]).total_seconds()
            st.metric("⏱️ Last RT", f"{time_since:.0f}s ago")
        else:
            st.metric("⏱️ Last RT", "Never")


def create_optimized_chart(df: pd.DataFrame, chart_func, title: str):
    """Create an optimized Plotly chart with consistent styling"""
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


def create_speed_chart(df: pd.DataFrame):
    """Generate a line chart showing vehicle speed over time"""
    if df.empty or "speed_ms" not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = px.line(
        df, x="timestamp", y="speed_ms",
        title="🚗 Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4"],
    )
    return fig


def create_power_chart(df: pd.DataFrame):
    """Generate a subplot chart displaying electrical system performance"""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["voltage_v"], name="Voltage (V)",
                  line=dict(color="#2ca02c", width=2)), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["current_a"], name="Current (A)",
                  line=dict(color="#d62728", width=2)), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["power_w"], name="Power (W)",
                  line=dict(color="#ff7f0e", width=2)), row=2, col=1,
    )

    fig.update_layout(height=500, title_text="⚡ Electrical System Performance")
    return fig


def create_imu_chart(df: pd.DataFrame):
    """Generate IMU data visualization"""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
        vertical_spacing=0.25,
    )

    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                      line=dict(color=colors_gyro[i], width=2)), row=1, col=1,
        )

    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                      line=dict(color=colors_accel[i], width=2)), row=2, col=1,
        )

    fig.update_layout(height=600, title_text="🎮 IMU Sensor Data Analysis")
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Generate efficiency analysis chart"""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = px.scatter(
        df, x="speed_ms", y="power_w",
        color="voltage_v" if "voltage_v" in df.columns else None,
        title="⚡ Efficiency Analysis: Speed vs Power Consumption",
        labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
        color_continuous_scale="viridis",
    )
    return fig


def create_gps_map(df: pd.DataFrame):
    """Generate GPS tracking map"""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    center_point = dict(lat=df_valid["latitude"].mean(), lon=df_valid["longitude"].mean())

    fig = px.scatter_map(
        df_valid, lat="latitude", lon="longitude",
        color="speed_ms" if "speed_ms" in df_valid.columns else None,
        size="power_w" if "power_w" in df_valid.columns else None,
        hover_data=["speed_ms", "power_w", "voltage_v"] if all(col in df_valid.columns for col in ["speed_ms", "power_w", "voltage_v"]) else None,
        map_style="open-street-map",
        title="🛰️ Vehicle Track and Performance",
        height=400, zoom=15, center=center_point,
        color_continuous_scale="plasma",
    )
    return fig


def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns suitable for plotting"""
    if df.empty:
        return []
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create customizable chart based on user configuration"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    if not y_col or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    try:
        if chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#ff7f0e"])
        elif chart_type == "bar":
            recent_df = df.tail(20)
            fig = px.bar(recent_df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#2ca02c"])
        elif chart_type == "histogram":
            fig = px.histogram(df, x=y_col, title=f"Distribution of {y_col}", color_discrete_sequence=["#d62728"])
        elif chart_type == "heatmap":
            numeric_cols = get_available_columns(df)
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title=f"🔥 Correlation Heatmap",
                              color_continuous_scale="RdBu_r", aspect="auto")
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])

        fig.update_layout(height=400)
        return fig

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )


def render_dynamic_charts_section(df: pd.DataFrame):
    """Render the dynamic charts creation section"""
    st.markdown("""
    <div class="instructions-container">
        <div class="instructions-title">🎯 Create Custom Charts</div>
        <div class="instructions-content">
            <p>Create custom visualizations with your preferred variables and chart types.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []

    if not available_columns:
        st.warning("⏳ No numeric data available for creating charts.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn"):
            try:
                new_chart = {
                    "id": str(uuid.uuid4()),
                    "title": "New Chart",
                    "chart_type": "line",
                    "x_axis": "timestamp" if "timestamp" in df.columns else available_columns[0],
                    "y_axis": available_columns[0] if available_columns else None,
                }
                st.session_state.dynamic_charts.append(new_chart)
                st.rerun()
            except Exception as e:
                st.error(f"Error adding chart: {e}")

    with col2:
        if st.session_state.dynamic_charts:
            st.success(f"📈 {len(st.session_state.dynamic_charts)} custom chart(s) active")

    if st.session_state.dynamic_charts:
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            try:
                with st.container(border=True):
                    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])

                    with col1:
                        new_title = st.text_input(
                            "Title", value=chart_config.get("title", "New Chart"),
                            key=f"title_{chart_config['id']}"
                        )
                        if new_title != chart_config.get("title"):
                            st.session_state.dynamic_charts[i]["title"] = new_title

                    with col2:
                        new_type = st.selectbox(
                            "Type", options=["line", "scatter", "bar", "histogram", "heatmap"],
                            index=["line", "scatter", "bar", "histogram", "heatmap"].index(
                                chart_config.get("chart_type", "line")
                            ),
                            key=f"type_{chart_config['id']}"
                        )
                        if new_type != chart_config.get("chart_type"):
                            st.session_state.dynamic_charts[i]["chart_type"] = new_type

                    with col3:
                        if chart_config.get("chart_type", "line") not in ["histogram", "heatmap"]:
                            x_options = (["timestamp"] + available_columns if "timestamp" in df.columns else available_columns)
                            current_x = chart_config.get("x_axis", x_options[0])
                            if current_x not in x_options and x_options:
                                current_x = x_options[0]

                            if x_options:
                                new_x = st.selectbox(
                                    "X-Axis", options=x_options,
                                    index=x_options.index(current_x) if current_x in x_options else 0,
                                    key=f"x_{chart_config['id']}"
                                )
                                if new_x != chart_config.get("x_axis"):
                                    st.session_state.dynamic_charts[i]["x_axis"] = new_x

                    with col4:
                        if chart_config.get("chart_type", "line") != "heatmap":
                            if available_columns:
                                current_y = chart_config.get("y_axis", available_columns[0])
                                if current_y not in available_columns:
                                    current_y = available_columns[0]

                                new_y = st.selectbox(
                                    "Y-Axis", options=available_columns,
                                    index=available_columns.index(current_y) if current_y in available_columns else 0,
                                    key=f"y_{chart_config['id']}"
                                )
                                if new_y != chart_config.get("y_axis"):
                                    st.session_state.dynamic_charts[i]["y_axis"] = new_y

                    with col5:
                        if st.button("🗑️", key=f"delete_{chart_config['id']}"):
                            try:
                                st.session_state.dynamic_charts.pop(i)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")

                    try:
                        if chart_config.get("chart_type") == "heatmap" or chart_config.get("y_axis"):
                            fig = create_dynamic_chart(df, chart_config)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config['id']}")
                        else:
                            st.warning("Please select a Y-axis variable for this chart.")
                    except Exception as e:
                        st.error(f"Error creating chart: {e}")

            except Exception as e:
                st.error(f"Error rendering chart {i}: {e}")


def main():
    """Main dashboard function"""
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard v0.6</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: var(--text-secondary);">Real-time + Historical Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    initialize_session_state()

    # Initialize Supabase manager
    if not st.session_state.supabase_manager:
        st.session_state.supabase_manager = SupabaseManager()

    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Connection Control")

        # Connection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Connect RT", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    time.sleep(2)

                with st.spinner("Connecting to real-time..."):
                    st.session_state.subscriber = TelemetrySubscriber()
                    if st.session_state.subscriber.connect():
                        st.success("✅ Real-time Connected!")
                    else:
                        st.error("❌ Real-time Failed!")
                st.rerun()

        with col2:
            if st.button("🛑 Disconnect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    st.session_state.subscriber = None
                st.info("🛑 Disconnected")
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

        render_connection_status(
            st.session_state.subscriber, 
            st.session_state.supabase_manager, 
            stats
        )

        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")

        st.divider()

        # Data source selection
        st.subheader("📊 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["combined", "realtime", "historical"],
            format_func=lambda x: {
                "combined": "🔄 Combined (RT + DB)",
                "realtime": "🔴 Real-time Only",
                "historical": "📊 Historical Only"
            }[x],
            key="data_source_radio"
        )
        st.session_state.data_source = data_source

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("RT Refresh Rate (s)", 1, 10, 3)

        st.info(f"📡 RT Channel: {CHANNEL_NAME}")
        st.info(f"💾 DB Updates: Every {DB_REFRESH_INTERVAL}s")

    # Session selection (only for historical/combined modes)
    if st.session_state.data_source in ["historical", "combined"]:
        selected_session = render_session_selector()
        st.session_state.selected_session = selected_session

    # Data ingestion and processing
    new_messages_count = 0
    
    # Real-time data ingestion
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        new_messages = st.session_state.subscriber.get_messages()
        if new_messages:
            new_messages_count = len(new_messages)
            new_df = pd.DataFrame(new_messages)
            
            if "timestamp" in new_df.columns:
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
            
            # Add calculated fields
            if all(col in new_df.columns for col in ["accel_x", "accel_y", "accel_z"]):
                new_df["total_acceleration"] = np.sqrt(
                    new_df["accel_x"]**2 + new_df["accel_y"]**2 + new_df["accel_z"]**2
                )
            
            if st.session_state.telemetry_data.empty:
                st.session_state.telemetry_data = new_df
            else:
                st.session_state.telemetry_data = pd.concat(
                    [st.session_state.telemetry_data, new_df], ignore_index=True
                )
            
            # Limit real-time data size
            if len(st.session_state.telemetry_data) > MAX_REALTIME_DATAPOINTS:
                st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_REALTIME_DATAPOINTS)
            
            st.session_state.last_update = datetime.now()

    # Database data refresh (every 10 seconds)
    current_time = datetime.now()
    if (current_time - st.session_state.last_db_update).total_seconds() >= DB_REFRESH_INTERVAL:
        if st.session_state.data_source in ["historical", "combined"]:
            if st.session_state.selected_session:
                # Load specific session data
                session_data = st.session_state.supabase_manager.get_session_data(
                    st.session_state.selected_session['session_id']
                )
                st.session_state.historical_data = session_data
            else:
                # Load recent data
                recent_data = st.session_state.supabase_manager.get_recent_data(60)  # Last 60 minutes
                st.session_state.historical_data = recent_data
        
        st.session_state.last_db_update = current_time
        st.session_state.db_refresh_counter += 1

    # Combine data based on selected source
    if st.session_state.data_source == "realtime":
        df = st.session_state.telemetry_data.copy()
        data_source_label = "realtime"
    elif st.session_state.data_source == "historical":
        df = st.session_state.historical_data.copy()
        data_source_label = "historical"
    else:  # combined
        dfs_to_combine = []
        if not st.session_state.telemetry_data.empty:
            dfs_to_combine.append(st.session_state.telemetry_data)
        if not st.session_state.historical_data.empty:
            dfs_to_combine.append(st.session_state.historical_data)
        
        if dfs_to_combine:
            df = pd.concat(dfs_to_combine, ignore_index=True)
            # Remove duplicates based on timestamp and session_id
            if 'timestamp' in df.columns:
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            df = pd.DataFrame()
        
        data_source_label = "combined"

    st.session_state.combined_data = df

    # Display data status
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Getting Started:**\n"
                "1. Ensure m1.py bridge is running\n"
                "2. Click 'Connect RT' for real-time data\n"
                "3. Select a session for historical data"
            )
        
        with col2:
            with st.expander("🔍 Debug Information"):
                st.json({
                    "RT Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                    "DB Connected": st.session_state.supabase_manager.client is not None,
                    "RT Messages": stats["messages_received"],
                    "DB Refreshes": st.session_state.db_refresh_counter,
                    "Selected Session": st.session_state.selected_session['session_id'][:8] + "..." if st.session_state.selected_session else None,
                    "Data Source": st.session_state.data_source,
                })
        return

    # Status row
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points")
    with col2:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col3:
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")
    with col4:
        st.info(f"🔄 DB: {st.session_state.db_refresh_counter}")

    # Calculate KPIs
    kpis = calculate_kpis(df)

    # Main dashboard tabs
    st.subheader("📈 Dashboard")
    
    tab_names = [
        "📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", 
        "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data"
    ]
    tabs = st.tabs(tab_names)

    # Overview tab
    with tabs[0]:
        render_kpi_header(kpis, data_source_label)
        
        # Add data composition info for combined mode
        if st.session_state.data_source == "combined":
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔴 Real-time Points", len(st.session_state.telemetry_data))
            with col2:
                st.metric("💾 Historical Points", len(st.session_state.historical_data))

    # Speed tab
    with tabs[1]:
        render_kpi_header(kpis, data_source_label)
        fig = create_optimized_chart(df, create_speed_chart, "Speed Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Power tab
    with tabs[2]:
        render_kpi_header(kpis, data_source_label)
        fig = create_optimized_chart(df, create_power_chart, "Power Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # IMU tab
    with tabs[3]:
        render_kpi_header(kpis, data_source_label)
        fig = create_optimized_chart(df, create_imu_chart, "IMU Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Efficiency tab
    with tabs[4]:
        render_kpi_header(kpis, data_source_label)
        fig = create_optimized_chart(df, create_efficiency_chart, "Efficiency Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # GPS tab
    with tabs[5]:
        render_kpi_header(kpis, data_source_label)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Custom tab
    with tabs[6]:
        render_kpi_header(kpis, data_source_label)
        render_dynamic_charts_section(df)

    # Data tab
    with tabs[7]:
        render_kpi_header(kpis, data_source_label)
        
        st.subheader("📃 Raw Telemetry Data")
        st.warning("ℹ️ Only the **last 100 datapoints** are displayed below. Download the CSV for the complete dataset.")
        
        # Show data composition
        if st.session_state.data_source == "combined" and 'session_id' in df.columns:
            st.info(f"📊 Data from {df['session_id'].nunique()} session(s)")
        
        st.dataframe(df.tail(100), use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Auto-refresh logic
    if (st.session_state.auto_refresh and 
        st.session_state.subscriber and 
        st.session_state.subscriber.is_connected):
        time.sleep(refresh_interval)
        st.rerun()

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>"
        "<p><strong>Shell Eco-marathon Telemetry Dashboard v0.6</strong> | Real-time + Historical Data Analysis</p>"
        "<p>🚗 Enhanced with session management and persistent storage</p>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
