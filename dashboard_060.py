# dashboard_updated.py - Enhanced Dashboard with Supabase Integration and Session Management
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
from typing import Dict, Any, List, Optional, Tuple
import threading
import queue
import asyncio
import uuid
from collections import defaultdict

# Disables tracemalloc warnings that can appear in Streamlit environments.
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Handles library imports with error checking.
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

# Function to set up terminal logging.
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

# Initializes terminal logging at application start.
setup_terminal_logging()

# Global configuration variables
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"

# Supabase configuration
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"

# Data source modes
DATA_SOURCE_REALTIME = "Real Time + Recent Data"
DATA_SOURCE_HISTORICAL = "Historical Database"

# Configures the Streamlit page for title, icon, layout, and initial sidebar state.
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

# Applies custom CSS for dashboard styling, including theme-aware colors and layout adjustments.
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

    .session-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 2px solid var(--border-color);
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .session-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
    }

    .session-card.selected {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }

    .data-source-selector {
        background: var(--bg-primary);
        border-radius: 8px;
        padding: 1rem;
        border: 2px solid var(--border-color);
        margin: 1rem 0;
    }

    .triangulation-info {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--warning-color);
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

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


class EnhancedTelemetryManager:
    """Enhanced telemetry manager with dual data sources and data triangulation."""

    def __init__(self):
        self.ably_subscriber = None
        self.supabase_client = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.connection_thread = None
        self.stats = {
            "messages_received": 0,
            "last_message_time": None,
            "connection_attempts": 0,
            "errors": 0,
            "last_error": None,
            "data_source_stats": {
                "ably_realtime": 0,
                "supabase_recent": 0,
                "streamlit_history": 0,
                "duplicates_removed": 0,
            }
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self.logger = logging.getLogger("EnhancedTelemetryManager")

    def connect_supabase(self) -> bool:
        """Initialize Supabase client connection"""
        try:
            self.supabase_client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
            self.logger.info("✅ Connected to Supabase database")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Supabase: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return False

    def connect_ably(self) -> bool:
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
            self.logger.error(f"❌ Ably connection failed: {e}")
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
            self.ably_subscriber = AblyRealtime(ABLY_API_KEY)

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

            self.ably_subscriber.connection.on("connected", on_connected)
            self.ably_subscriber.connection.on("disconnected", on_disconnected)
            self.ably_subscriber.connection.on("failed", on_failed)
            self.ably_subscriber.connection.on("suspended", on_disconnected)

            await self.ably_subscriber.connection.once_async("connected")

            channel = self.ably_subscriber.channels.get(CHANNEL_NAME)
            await channel.subscribe("telemetry_update", self._on_message_received)

            self.logger.info("✅ Successfully subscribed to Ably messages!")

            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)
                if hasattr(self.ably_subscriber.connection, "state"):
                    state = self.ably_subscriber.connection.state
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
            data = message.data
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON decode error: {e}")
                    return

            if not isinstance(data, dict):
                self.logger.warning(f"⚠️ Invalid data type: {type(data)}")
                return

            # Add data source tag
            data['data_source_type'] = 'ably_realtime'

            with self._lock:
                if self.message_queue.qsize() > 200:
                    while self.message_queue.qsize() > 100:
                        try:
                            self.message_queue.get_nowait()
                        except queue.Empty:
                            break

                self.message_queue.put(data)
                self.stats["messages_received"] += 1
                self.stats["data_source_stats"]["ably_realtime"] += 1
                self.stats["last_message_time"] = datetime.now()

        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Message error: {e}"

    def get_recent_supabase_data(self, minutes_back: int = 10) -> List[Dict[str, Any]]:
        """Get recent data from Supabase database"""
        try:
            if not self.supabase_client:
                return []

            # Calculate time threshold
            threshold_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)
            
            response = (
                self.supabase_client
                .table(SUPABASE_TABLE_NAME)
                .select("*")
                .gte("timestamp", threshold_time.isoformat())
                .order("timestamp", desc=True)
                .limit(1000)
                .execute()
            )

            data = response.data if response.data else []
            
            # Add data source tag
            for item in data:
                item['data_source_type'] = 'supabase_recent'
            
            with self._lock:
                self.stats["data_source_stats"]["supabase_recent"] += len(data)
            
            return data

        except Exception as e:
            self.logger.error(f"❌ Error fetching Supabase data: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Supabase fetch error: {e}"
            return []

    def get_historical_sessions(self) -> List[Dict[str, Any]]:
        """Get all available sessions from Supabase"""
        try:
            if not self.supabase_client:
                return []

            # Get session summary
            response = (
                self.supabase_client
                .table(SUPABASE_TABLE_NAME)
                .select("session_id, timestamp")
                .order("timestamp", desc=False)
                .execute()
            )

            if not response.data:
                return []

            # Group by session and calculate stats
            sessions = defaultdict(lambda: {
                "start_time": None,
                "end_time": None,
                "record_count": 0
            })

            for record in response.data:
                session_id = record["session_id"]
                timestamp = datetime.fromisoformat(record["timestamp"].replace('Z', '+00:00'))
                
                if sessions[session_id]["start_time"] is None or timestamp < sessions[session_id]["start_time"]:
                    sessions[session_id]["start_time"] = timestamp
                
                if sessions[session_id]["end_time"] is None or timestamp > sessions[session_id]["end_time"]:
                    sessions[session_id]["end_time"] = timestamp
                
                sessions[session_id]["record_count"] += 1

            # Format session list
            session_list = []
            for session_id, stats in sessions.items():
                duration = stats["end_time"] - stats["start_time"]
                session_list.append({
                    "session_id": session_id,
                    "start_time": stats["start_time"],
                    "end_time": stats["end_time"],
                    "duration": duration,
                    "record_count": stats["record_count"]
                })

            # Sort by start time (newest first)
            session_list.sort(key=lambda x: x["start_time"], reverse=True)
            return session_list

        except Exception as e:
            self.logger.error(f"❌ Error fetching sessions: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Session fetch error: {e}"
            return []

    def get_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all data for a specific session"""
        try:
            if not self.supabase_client:
                return []

            response = (
                self.supabase_client
                .table(SUPABASE_TABLE_NAME)
                .select("*")
                .eq("session_id", session_id)
                .order("timestamp", desc=False)
                .execute()
            )

            data = response.data if response.data else []
            
            # Add data source tag
            for item in data:
                item['data_source_type'] = 'supabase_historical'
            
            return data

        except Exception as e:
            self.logger.error(f"❌ Error fetching session data: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Session data fetch error: {e}"
            return []

    def triangulate_data(self, ably_data: List[Dict], supabase_data: List[Dict], streamlit_data: pd.DataFrame) -> pd.DataFrame:
        """Triangulate data from multiple sources to eliminate duplicates and ensure completeness"""
        try:
            all_data = []
            seen_records = set()  # Track unique records by timestamp + session_id
            
            # Add streamlit history data first (oldest source)
            if not streamlit_data.empty:
                for _, row in streamlit_data.iterrows():
                    row_dict = row.to_dict()
                    row_dict['data_source_type'] = 'streamlit_history'
                    
                    # Create unique key
                    timestamp = row_dict.get('timestamp', '')
                    session_id = row_dict.get('session_id', 'unknown')
                    message_id = row_dict.get('message_id', 0)
                    unique_key = f"{timestamp}_{session_id}_{message_id}"
                    
                    if unique_key not in seen_records:
                        all_data.append(row_dict)
                        seen_records.add(unique_key)
                        with self._lock:
                            self.stats["data_source_stats"]["streamlit_history"] += 1

            # Add Supabase data (medium priority)
            for record in supabase_data:
                timestamp = record.get('timestamp', '')
                session_id = record.get('session_id', 'unknown')
                message_id = record.get('message_id', 0)
                unique_key = f"{timestamp}_{session_id}_{message_id}"
                
                if unique_key not in seen_records:
                    all_data.append(record)
                    seen_records.add(unique_key)
                else:
                    with self._lock:
                        self.stats["data_source_stats"]["duplicates_removed"] += 1

            # Add Ably data (highest priority - most recent)
            for record in ably_data:
                timestamp = record.get('timestamp', '')
                session_id = record.get('session_id', 'unknown')
                message_id = record.get('message_id', 0)
                unique_key = f"{timestamp}_{session_id}_{message_id}"
                
                if unique_key not in seen_records:
                    all_data.append(record)
                    seen_records.add(unique_key)
                else:
                    with self._lock:
                        self.stats["data_source_stats"]["duplicates_removed"] += 1

            # Convert to DataFrame and ensure timestamp is properly formatted
            if all_data:
                df = pd.DataFrame(all_data)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                    df = df.sort_values("timestamp").reset_index(drop=True)
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ Error triangulating data: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Triangulation error: {e}"
            return pd.DataFrame()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all queued messages from Ably"""
        messages = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
        return messages

    def disconnect(self):
        """Disconnect from all sources"""
        try:
            self.logger.info("🛑 Disconnecting...")
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False

            if self.ably_subscriber:
                try:
                    self.ably_subscriber.close()
                    self.logger.info("✅ Ably connection closed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably: {e}")

            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=5)

            self.logger.info("🔚 Disconnection complete")

        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Disconnect error: {e}"
        finally:
            self.ably_subscriber = None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self._lock:
            return self.stats.copy()


def initialize_session_state():
    """Initializes Streamlit session state variables with default values if they don't exist."""
    defaults = {
        "telemetry_manager": None,
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "data_source_mode": DATA_SOURCE_REALTIME,
        "selected_session": None,
        "available_sessions": [],
        "last_session_refresh": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates key performance indicators from the telemetry DataFrame."""
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
            "energy_j", "speed_ms", "distance_m", "power_w", "total_acceleration",
            "gyro_x", "gyro_y", "gyro_z"
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
            kpis["efficiency_km_per_mj"] = (
                kpis["total_distance_km"] / kpis["total_energy_mj"]
            )

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


def render_kpi_header(kpis: Dict[str, float]):
    """Renders a compact performance dashboard at the top of a tab."""
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
    """Renders the Overview tab with enhanced KPI display."""
    st.markdown("### 📊 Performance Overview")
    st.markdown("Real-time key performance indicators for your Shell Eco-marathon vehicle")

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


def render_data_source_selector():
    """Renders the data source selection interface."""
    st.markdown('<div class="data-source-selector">', unsafe_allow_html=True)
    st.markdown("### 🔧 Data Source Configuration")
    
    # Data source mode selection
    new_mode = st.radio(
        "Select Data Source:",
        options=[DATA_SOURCE_REALTIME, DATA_SOURCE_HISTORICAL],
        index=0 if st.session_state.data_source_mode == DATA_SOURCE_REALTIME else 1,
        help="Choose between real-time data with recent history or historical session data"
    )
    
    if new_mode != st.session_state.data_source_mode:
        st.session_state.data_source_mode = new_mode
        st.rerun()

    # Show mode-specific information
    if st.session_state.data_source_mode == DATA_SOURCE_REALTIME:
        st.markdown(
            """
            **Real Time + Recent Data Mode:**
            - 🔄 Live telemetry from Ably real-time channel
            - 📊 Recent data from Supabase database (last 10 minutes)
            - 💾 Streamlit session history
            - 🔍 Automatic data triangulation to prevent duplicates
            """
        )
        
        # Display triangulation info
        if st.session_state.telemetry_manager:
            stats = st.session_state.telemetry_manager.get_stats()
            triangulation_stats = stats.get("data_source_stats", {})
            
            st.markdown('<div class="triangulation-info">', unsafe_allow_html=True)
            st.markdown("**📊 Data Triangulation Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔄 Ably Real-time", triangulation_stats.get("ably_realtime", 0))
            with col2:
                st.metric("📊 Supabase Recent", triangulation_stats.get("supabase_recent", 0))
            with col3:
                st.metric("💾 Streamlit History", triangulation_stats.get("streamlit_history", 0))
            with col4:
                st.metric("🗑️ Duplicates Removed", triangulation_stats.get("duplicates_removed", 0))
            st.markdown('</div>', unsafe_allow_html=True)

    else:  # Historical mode
        st.markdown(
            """
            **Historical Database Mode:**
            - 📚 Access complete historical sessions
            - 📅 Select specific sessions to analyze
            - 📈 Full dataset without real-time limitations
            """
        )
        
        # Refresh sessions button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Refresh Sessions"):
                if st.session_state.telemetry_manager and st.session_state.telemetry_manager.supabase_client:
                    st.session_state.available_sessions = st.session_state.telemetry_manager.get_historical_sessions()
                    st.session_state.last_session_refresh = datetime.now()
                    st.success("Sessions refreshed!")
                else:
                    st.error("Supabase not connected")

        # Display available sessions
        if st.session_state.available_sessions:
            st.markdown("**Available Sessions:**")
            
            for session in st.session_state.available_sessions:
                session_id = session["session_id"]
                start_time = session["start_time"]
                duration = session["duration"]
                record_count = session["record_count"]
                
                # Create a compact session card
                is_selected = st.session_state.selected_session == session_id
                card_class = "session-card selected" if is_selected else "session-card"
                
                with st.container():
                    if st.button(
                        f"📋 Session: {session_id[:8]}...\n"
                        f"📅 Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"⏱️ Duration: {duration}\n"
                        f"📊 Records: {record_count}",
                        key=f"session_{session_id}",
                        use_container_width=True
                    ):
                        st.session_state.selected_session = session_id
                        st.rerun()
        else:
            st.info("No historical sessions available. Click 'Refresh Sessions' to load data.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_connection_status(manager, stats):
    """Renders connection status and statistics in the sidebar."""
    if st.session_state.data_source_mode == DATA_SOURCE_REALTIME:
        if manager and manager.is_connected:
            st.sidebar.markdown(
                '<div class="status-indicator status-connected">✅ Real-time Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<div class="status-indicator status-disconnected">❌ Real-time Disconnected</div>',
                unsafe_allow_html=True,
            )
    else:
        if manager and manager.supabase_client:
            st.sidebar.markdown(
                '<div class="status-indicator status-connected">✅ Database Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<div class="status-indicator status-disconnected">❌ Database Disconnected</div>',
                unsafe_allow_html=True,
            )

    # Display statistics
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


# Import all the chart creation functions from the original dashboard
def create_optimized_chart(df: pd.DataFrame, chart_func, title: str):
    """Creates an optimized Plotly chart by applying consistent styling."""
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
    """Generates a line chart showing vehicle speed over time."""
    if df.empty or "speed_ms" not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = px.line(
        df, x="timestamp", y="speed_ms", title="🚗 Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4"],
    )
    return fig


def create_power_chart(df: pd.DataFrame):
    """Generates a subplot chart displaying voltage, current, and power output over time."""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
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
    """Generates a subplot chart for IMU data."""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
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


def create_imu_chart_2(df: pd.DataFrame):
    """Generates a detailed IMU chart with individual subplots."""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("🌀 Gyro X", "🌀 Gyro Y", "🌀 Gyro Z", "📊 Accel X", "📊 Accel Y", "📊 Accel Z"),
        vertical_spacing=0.3, horizontal_spacing=0.1,
    )

    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]

    for i, (axis, color) in enumerate(zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                      line=dict(color=color, width=2), showlegend=False), row=1, col=i + 1,
        )

    for i, (axis, color) in enumerate(zip(["accel_x", "accel_y", "accel_z"], accel_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                      line=dict(color=color, width=2), showlegend=False), row=2, col=i + 1,
        )

    fig.update_layout(height=600, title_text="🎮 Detailed IMU Sensor Analysis")
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Generates a scatter plot for efficiency analysis."""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig = px.scatter(
        df, x="speed_ms", y="power_w", color="voltage_v" if "voltage_v" in df.columns else None,
        title="⚡ Efficiency Analysis: Speed vs Power Consumption",
        labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
        color_continuous_scale="viridis",
    )
    return fig


def create_gps_map(df: pd.DataFrame):
    """Generates a scatter map to display vehicle GPS tracking."""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    center_point = dict(lat=df_valid["latitude"].mean(), lon=df_valid["longitude"].mean())

    fig = px.scatter_map(
        df_valid, lat="latitude", lon="longitude",
        color="speed_ms" if "speed_ms" in df_valid.columns else None,
        size="power_w" if "power_w" in df_valid.columns else None,
        hover_data=["speed_ms", "power_w", "voltage_v"] if all(
            col in df_valid.columns for col in ["speed_ms", "power_w", "voltage_v"]
        ) else None,
        map_style="open-street-map", title="🛰️ Vehicle Track and Performance",
        height=400, zoom=15, center=center_point, color_continuous_scale="plasma",
    )
    return fig


def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Retrieves a list of numeric columns from the DataFrame suitable for plotting."""
    if df.empty:
        return []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Creates a customizable chart based on user-defined configurations."""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    if not y_col or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
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
                fig = px.imshow(corr_matrix, title=f"🔥 Correlation Heatmap", color_continuous_scale="RdBu_r", aspect="auto")
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])

        fig.update_layout(height=400)
        return fig

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )


def render_dynamic_charts_section(df: pd.DataFrame):
    """Renders the section for creating and displaying dynamic, user-configured charts."""
    st.markdown(
        """
    <div class="instructions-container">
        <div class="instructions-title">🎯 Create Custom Charts</div>
        <div class="instructions-content">
            <p>Click <strong>"Add Chart"</strong> to create custom visualizations with your preferred variables and chart types.</p>
        </div>
    </div>
    """, unsafe_allow_html=True,
    )

    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []

    if not available_columns:
        st.warning("⏳ No numeric data available for creating charts. Load data first.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart"):
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
                            key=f"title_{chart_config['id']}",
                        )
                        if new_title != chart_config.get("title"):
                            st.session_state.dynamic_charts[i]["title"] = new_title

                    with col2:
                        new_type = st.selectbox(
                            "Type", options=["line", "scatter", "bar", "histogram", "heatmap"],
                            index=["line", "scatter", "bar", "histogram", "heatmap"].index(
                                chart_config.get("chart_type", "line")
                            ), key=f"type_{chart_config['id']}",
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
                                    key=f"x_{chart_config['id']}",
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
                                    key=f"y_{chart_config['id']}",
                                )
                                if new_y != chart_config.get("y_axis"):
                                    st.session_state.dynamic_charts[i]["y_axis"] = new_y

                    with col5:
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"):
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
    """Main dashboard function, managing UI elements, data ingestion, and chart rendering."""
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    initialize_session_state()

    # Initialize telemetry manager if not already done
    if st.session_state.telemetry_manager is None:
        st.session_state.telemetry_manager = EnhancedTelemetryManager()
        if st.session_state.telemetry_manager.connect_supabase():
            st.session_state.available_sessions = st.session_state.telemetry_manager.get_historical_sessions()

    manager = st.session_state.telemetry_manager

    # Render sidebar elements
    with st.sidebar:
        st.header("🔧 Connection Control")

        # Connection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", use_container_width=True):
                if manager:
                    if st.session_state.data_source_mode == DATA_SOURCE_REALTIME:
                        manager.disconnect()
                        time.sleep(2)
                        with st.spinner("Connecting to real-time..."):
                            if manager.connect_ably() and manager.connect_supabase():
                                st.success("✅ Connected!")
                            else:
                                st.error("❌ Failed!")
                    else:
                        with st.spinner("Connecting to database..."):
                            if manager.connect_supabase():
                                st.session_state.available_sessions = manager.get_historical_sessions()
                                st.success("✅ Connected!")
                            else:
                                st.error("❌ Failed!")
                st.rerun()

        with col2:
            if st.button("🛑 Disconnect", use_container_width=True):
                if manager:
                    manager.disconnect()
                st.info("🛑 Disconnected")
                st.rerun()

        # Display connection status
        stats = manager.get_stats() if manager else {
            "messages_received": 0, "connection_attempts": 0, "errors": 0,
            "last_message_time": None, "last_error": None,
        }

        render_connection_status(manager, stats)

        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")

        st.divider()

        # Data source selector
        render_data_source_selector()

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        new_auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

        if new_auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = new_auto_refresh

        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)

    # Data ingestion based on selected mode
    new_messages_count = 0
    
    if st.session_state.data_source_mode == DATA_SOURCE_REALTIME:
        # Real-time + Recent data mode
        if manager and (manager.is_connected or manager.supabase_client):
            # Get data from multiple sources
            ably_messages = manager.get_messages() if manager.is_connected else []
            supabase_data = manager.get_recent_supabase_data() if manager.supabase_client else []
            
            # Triangulate data
            triangulated_df = manager.triangulate_data(
                ably_messages, supabase_data, st.session_state.telemetry_data
            )
            
            if not triangulated_df.empty:
                new_messages_count = len(triangulated_df) - len(st.session_state.telemetry_data)
                st.session_state.telemetry_data = triangulated_df
                st.session_state.last_update = datetime.now()

    else:
        # Historical database mode
        if st.session_state.selected_session and manager and manager.supabase_client:
            session_data = manager.get_session_data(st.session_state.selected_session)
            if session_data:
                st.session_state.telemetry_data = pd.DataFrame(session_data)
                if "timestamp" in st.session_state.telemetry_data.columns:
                    st.session_state.telemetry_data["timestamp"] = pd.to_datetime(
                        st.session_state.telemetry_data["timestamp"]
                    )
                st.session_state.last_update = datetime.now()

    df = st.session_state.telemetry_data.copy()

    # Display empty state message if no data
    if df.empty:
        if st.session_state.data_source_mode == DATA_SOURCE_REALTIME:
            st.warning("⏳ Waiting for telemetry data...")
            col1, col2 = st.columns(2)
            with col1:
                st.info(
                    "**Getting Started:**\n"
                    "1. Ensure m1.py bridge is running\n"
                    "2. Click 'Connect' to start receiving data\n"
                    "3. Data will be triangulated from multiple sources"
                )
            with col2:
                with st.expander("🔍 Debug Information"):
                    st.json({
                        "Ably Connected": manager.is_connected if manager else False,
                        "Supabase Connected": bool(manager.supabase_client) if manager else False,
                        "Messages": stats["messages_received"],
                        "Errors": stats["errors"],
                        "Data Source Mode": st.session_state.data_source_mode,
                    })
        else:
            st.warning("📚 Select a historical session to view data")
            if not st.session_state.available_sessions:
                st.info("Click 'Refresh Sessions' in the sidebar to load available sessions.")

        return

    # Display status information
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points loaded")
    with col2:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col3:
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")

    # Calculate KPIs
    kpis = calculate_kpis(df)

    # Display main dashboard tabs
    st.subheader("📈 Dashboard")

    tab_names = [
        "📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", "🎮 IMU Detail",
        "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data",
    ]
    tabs = st.tabs(tab_names)

    # Render each tab
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
        fig = create_optimized_chart(df, create_imu_chart_2, "IMU Detail Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_efficiency_chart, "Efficiency Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tabs[7]:
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)

    with tabs[8]:
        render_kpi_header(kpis)
        st.subheader("📃 Raw Telemetry Data")
        
        # Display only last 100 records in the table (as requested)
        display_df = df.tail(100)
        st.warning(f"ℹ️ Showing the **last {len(display_df)} datapoints** in the table below. Download CSV for complete dataset.")
        st.dataframe(display_df, use_container_width=True, height=400)

        # Download button for complete dataset (no limit)
        csv = df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Complete Dataset ({len(df):,} records)",
            data=csv,
            file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Show data source breakdown
        if 'data_source_type' in df.columns:
            st.subheader("📊 Data Source Breakdown")
            source_counts = df['data_source_type'].value_counts()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔄 Ably Real-time", source_counts.get('ably_realtime', 0))
            with col2:
                st.metric("📊 Supabase Recent", source_counts.get('supabase_recent', 0))
            with col3:
                st.metric("💾 Streamlit History", source_counts.get('streamlit_history', 0))

    # Auto-refresh implementation
    if (st.session_state.auto_refresh and st.session_state.data_source_mode == DATA_SOURCE_REALTIME 
        and manager and (manager.is_connected or manager.supabase_client)):
        time.sleep(refresh_interval)
        st.rerun()

    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>"
        "<p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | Enhanced with Multi-Source Data Integration</p>"
        "<p>🚗 Real-time + Historical Analysis | 🔍 Data Triangulation | 📊 Session Management</p>"
        "</div>", unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
