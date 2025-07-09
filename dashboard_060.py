# dashboard_enhanced.py - Enhanced Dashboard with Supabase Integration and Session Management
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
import warnings

# Disables tracemalloc warnings that can appear in Streamlit environments.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Handles library imports with error checking.
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("❌ Ably library not available. Please install: pip install ably")
    st.stop()

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.error("❌ Supabase library not available. Please install: pip install supabase")
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
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"
MAX_DATAPOINTS = 1_000_000_000  # 1 billion data points limit

# Configures the Streamlit page
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Shell Eco-marathon Real-time Telemetry Dashboard with Historical Data",
    },
)

# Enhanced CSS with modern styling for both light and dark modes
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
        --shadow-light: rgba(0,0,0,0.1);
        --shadow-medium: rgba(0,0,0,0.15);
    }

    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #fafafa;
            --text-secondary: #a0a0a0;
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --border-color: #4a4a4a;
            --shadow-light: rgba(255,255,255,0.1);
            --shadow-medium: rgba(255,255,255,0.15);
        }
    }

    .main-header {
        font-size: 2.2rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 0 2px 4px var(--shadow-light);
    }

    .status-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
    }

    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3);
    }

    .status-connecting {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 2px solid #ffc107;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3);
    }

    .status-historical {
        background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
        color: #383d41;
        border: 2px solid #6c757d;
        box-shadow: 0 4px 12px rgba(108, 117, 125, 0.3);
    }

    .session-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid var(--primary-color);
        box-shadow: 0 6px 20px var(--shadow-medium);
        transition: all 0.3s ease;
    }

    .session-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-medium);
    }

    .session-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .session-detail {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: var(--text-primary);
    }

    .data-source-selector {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: 0 4px 16px var(--shadow-light);
    }

    .historical-notice {
        background: linear-gradient(135deg, #e9ecef 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--info-color);
        color: var(--text-primary);
        font-weight: 500;
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
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .chart-type-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow-medium);
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

    /* Enhanced button styling */
    .stButton > button {
        border-radius: 12px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .stButton > button:hover {
        background: transparent;
        color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4);
    }

    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 50;
        background: var(--bg-primary);
        border-bottom: 2px solid var(--border-color);
        border-radius: 12px 12px 0 0;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .chart-type-grid {
            grid-template-columns: 1fr;
        }
        .session-info {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


class EnhancedTelemetrySubscriber:
    """Enhanced telemetry subscriber with Ably Realtime and history support."""

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

        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Message error: {e}"

    async def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history from Ably channel"""
        try:
            if not self.channel:
                return []
            
            history_result = await self.channel.history(limit=min(limit, 1000))
            messages = []
            
            for message in history_result.items:
                if isinstance(message.data, dict):
                    messages.append(message.data)
                elif isinstance(message.data, str):
                    try:
                        messages.append(json.loads(message.data))
                    except json.JSONDecodeError:
                        continue
            
            return messages
        except Exception as e:
            self.logger.error(f"❌ Error getting history: {e}")
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


class SupabaseManager:
    """Manages Supabase database connections and queries."""

    def __init__(self):
        self.client = None
        self.logger = logging.getLogger("SupabaseManager")

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

            response = self.client.table(SUPABASE_TABLE_NAME)\
                .select("session_id, timestamp")\
                .order("timestamp", desc=False)\
                .execute()

            if not response.data:
                return []

            # Group by session_id and get session info
            sessions = {}
            for record in response.data:
                session_id = record['session_id']
                timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session_id': session_id,
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'record_count': 1
                    }
                else:
                    sessions[session_id]['end_time'] = max(sessions[session_id]['end_time'], timestamp)
                    sessions[session_id]['record_count'] += 1

            # Calculate duration for each session
            for session in sessions.values():
                session['duration'] = session['end_time'] - session['start_time']

            return list(sessions.values())

        except Exception as e:
            self.logger.error(f"❌ Error getting sessions: {e}")
            return []

    def get_session_data(self, session_id: str, limit: int = None) -> pd.DataFrame:
        """Get telemetry data for a specific session"""
        try:
            if not self.client:
                return pd.DataFrame()

            query = self.client.table(SUPABASE_TABLE_NAME)\
                .select("*")\
                .eq("session_id", session_id)\
                .order("timestamp", desc=False)

            if limit:
                query = query.limit(limit)

            response = query.execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            self.logger.error(f"❌ Error getting session data: {e}")
            return pd.DataFrame()

    def get_current_session_data(self, session_id: str, since: datetime = None) -> pd.DataFrame:
        """Get current session data, optionally since a specific time"""
        try:
            if not self.client:
                return pd.DataFrame()

            query = self.client.table(SUPABASE_TABLE_NAME)\
                .select("*")\
                .eq("session_id", session_id)\
                .order("timestamp", desc=False)

            if since:
                query = query.gte("timestamp", since.isoformat())

            response = query.execute()

            if not response.data:
                return pd.DataFrame()

            df = pd.DataFrame(response.data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            self.logger.error(f"❌ Error getting current session data: {e}")
            return pd.DataFrame()


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        "subscriber": None,
        "supabase_manager": None,
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "data_source": "realtime_session",  # "realtime_session" or "historical"
        "selected_session": None,
        "available_sessions": [],
        "is_historical_mode": False,
        "last_db_sync": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def merge_data_sources(realtime_data: pd.DataFrame, 
                      history_data: pd.DataFrame, 
                      db_data: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently merge data from multiple sources using timestamp triangulation
    to ensure no duplicates and no data loss.
    """
    try:
        all_dataframes = []
        
        # Add source tags to track data origin
        if not realtime_data.empty:
            realtime_data = realtime_data.copy()
            realtime_data['data_origin'] = 'realtime'
            all_dataframes.append(realtime_data)
        
        if not history_data.empty:
            history_data = history_data.copy()
            history_data['data_origin'] = 'ably_history'
            all_dataframes.append(history_data)
        
        if not db_data.empty:
            db_data = db_data.copy()
            db_data['data_origin'] = 'database'
            all_dataframes.append(db_data)
        
        if not all_dataframes:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        if combined_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        # Remove duplicates based on timestamp and message_id (if available)
        if 'message_id' in combined_df.columns:
            # Use both timestamp and message_id for deduplication
            combined_df = combined_df.drop_duplicates(
                subset=['timestamp', 'message_id'], 
                keep='first'
            )
        else:
            # Use timestamp with a small tolerance for deduplication
            combined_df = combined_df.drop_duplicates(
                subset=['timestamp'], 
                keep='first'
            )
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Limit to MAX_DATAPOINTS
        if len(combined_df) > MAX_DATAPOINTS:
            combined_df = combined_df.tail(MAX_DATAPOINTS)
        
        return combined_df
        
    except Exception as e:
        st.error(f"Error merging data sources: {e}")
        return pd.DataFrame()


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


def render_kpi_header(kpis: Dict[str, float]):
    """Render KPI header with metrics"""
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
    st.markdown("### 📋 Session Selection")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        options=["realtime_session", "historical"],
        format_func=lambda x: {
            "realtime_session": "🔴 Real-Time + Session Data",
            "historical": "📚 Historical Data"
        }[x],
        key="data_source_selector",
        help="Real-Time + Session: Current session with live updates\nHistorical: Past session data from database"
    )
    
    st.session_state.data_source = data_source
    st.session_state.is_historical_mode = (data_source == "historical")
    
    if data_source == "historical":
        # Show historical data notice
        st.markdown(
            """
            <div class="historical-notice">
                📚 <strong>Historical Mode Active</strong><br>
                You are viewing past session data. Auto-refresh is disabled in this mode.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Session selection for historical data
        if st.session_state.available_sessions:
            session_options = []
            for session in st.session_state.available_sessions:
                session_id_short = session['session_id'][:8] + "..."
                start_time = session['start_time'].strftime('%Y-%m-%d %H:%M:%S')
                duration = str(session['duration']).split('.')[0]  # Remove microseconds
                records = session['record_count']
                
                option_text = f"📋 {session_id_short} | 📅 {start_time} | ⏱️ {duration} | 📊 {records} records"
                session_options.append((option_text, session['session_id']))
            
            if session_options:
                selected_option = st.selectbox(
                    "Choose a session:",
                    options=[opt[0] for opt in session_options],
                    key="session_selector"
                )
                
                # Find the selected session ID
                selected_session_id = None
                for opt_text, session_id in session_options:
                    if opt_text == selected_option:
                        selected_session_id = session_id
                        break
                
                st.session_state.selected_session = selected_session_id
                
                # Display selected session details
                if selected_session_id:
                    selected_session_info = next(
                        (s for s in st.session_state.available_sessions 
                         if s['session_id'] == selected_session_id), 
                        None
                    )
                    
                    if selected_session_info:
                        st.markdown(
                            f"""
                            <div class="session-card">
                                <h4>Selected Session Details</h4>
                                <div class="session-info">
                                    <div class="session-detail">
                                        📋 <strong>Session:</strong> {selected_session_info['session_id'][:8]}...
                                    </div>
                                    <div class="session-detail">
                                        📅 <strong>Start:</strong> {selected_session_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
                                    </div>
                                    <div class="session-detail">
                                        ⏱️ <strong>Duration:</strong> {str(selected_session_info['duration']).split('.')[0]}
                                    </div>
                                    <div class="session-detail">
                                        📊 <strong>Records:</strong> {selected_session_info['record_count']:,}
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.warning("No sessions available for selection.")
        else:
            st.info("Loading available sessions...")
    
    else:
        # Real-time + Session mode
        st.markdown(
            """
            <div class="data-source-selector">
                <h4>🔴 Real-Time + Session Data Mode</h4>
                <p>This mode combines:</p>
                <ul>
                    <li><strong>Supabase Database:</strong> All past data from current session</li>
                    <li><strong>Ably History:</strong> Recent message history (up to 100 messages)</li>
                    <li><strong>Real-Time Stream:</strong> Live incoming data</li>
                </ul>
                <p>Data is intelligently merged to prevent duplicates and ensure completeness.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_connection_status(subscriber, supabase_manager, stats):
    """Render connection status in sidebar"""
    st.sidebar.markdown("### 🔌 Connection Status")
    
    # Ably connection status
    if subscriber and subscriber.is_connected:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Ably Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Ably Disconnected</div>',
            unsafe_allow_html=True,
        )
    
    # Supabase connection status
    if supabase_manager and supabase_manager.client:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Supabase Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Supabase Disconnected</div>',
            unsafe_allow_html=True,
        )
    
    # Historical mode indicator
    if st.session_state.is_historical_mode:
        st.sidebar.markdown(
            '<div class="status-indicator status-historical">📚 Historical Mode</div>',
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


# Chart creation functions (keeping the same as original)
def create_speed_chart(df: pd.DataFrame):
    """Generate speed chart"""
    if df.empty or "speed_ms" not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = px.line(
        df, x="timestamp", y="speed_ms", title="🚗 Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4"]
    )
    return fig


def create_power_chart(df: pd.DataFrame):
    """Generate power chart"""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["voltage_v"], name="Voltage (V)",
        line=dict(color="#2ca02c", width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["current_a"], name="Current (A)",
        line=dict(color="#d62728", width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["power_w"], name="Power (W)",
        line=dict(color="#ff7f0e", width=2)
    ), row=2, col=1)

    fig.update_layout(height=500, title_text="⚡ Electrical System Performance")
    return fig


def create_imu_chart(df: pd.DataFrame):
    """Generate IMU chart"""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
        vertical_spacing=0.25
    )

    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
            line=dict(color=colors_gyro[i], width=2)
        ), row=1, col=1)

    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
            line=dict(color=colors_accel[i], width=2)
        ), row=2, col=1)

    fig.update_layout(height=600, title_text="🎮 IMU Sensor Data Analysis")
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Generate efficiency chart"""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = px.scatter(
        df, x="speed_ms", y="power_w",
        color="voltage_v" if "voltage_v" in df.columns else None,
        title="⚡ Efficiency Analysis: Speed vs Power Consumption",
        labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
        color_continuous_scale="viridis"
    )
    return fig


def create_gps_map(df: pd.DataFrame):
    """Generate GPS map"""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
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
        height=400, zoom=15, center=center_point, color_continuous_scale="plasma"
    )
    return fig


def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get available numeric columns for plotting"""
    if df.empty:
        return []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create dynamic chart based on configuration"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    if not y_col or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
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
                fig = px.imshow(corr_matrix, title="🔥 Correlation Heatmap", 
                               color_continuous_scale="RdBu_r", aspect="auto")
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])

        fig.update_layout(height=400)
        return fig

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )


def render_dynamic_charts_section(df: pd.DataFrame):
    """Render dynamic charts section (keeping from original)"""
    st.markdown("### 🎯 Create Custom Charts")
    
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

    available_columns = get_available_columns(df)
    if not available_columns:
        st.warning("⏳ No numeric data available for creating charts.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn"):
            new_chart = {
                "id": str(uuid.uuid4()),
                "title": "New Chart",
                "chart_type": "line",
                "x_axis": "timestamp" if "timestamp" in df.columns else available_columns[0],
                "y_axis": available_columns[0] if available_columns else None,
            }
            st.session_state.dynamic_charts.append(new_chart)
            st.rerun()

    with col2:
        if st.session_state.dynamic_charts:
            st.success(f"📈 {len(st.session_state.dynamic_charts)} custom chart(s) active")

    # Render existing charts
    for i, chart_config in enumerate(st.session_state.dynamic_charts):
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])

            with col1:
                new_title = st.text_input(
                    "Title", value=chart_config.get("title", "New Chart"),
                    key=f"title_{chart_config['id']}"
                )
                st.session_state.dynamic_charts[i]["title"] = new_title

            with col2:
                new_type = st.selectbox(
                    "Type", options=["line", "scatter", "bar", "histogram", "heatmap"],
                    index=["line", "scatter", "bar", "histogram", "heatmap"].index(
                        chart_config.get("chart_type", "line")
                    ),
                    key=f"type_{chart_config['id']}"
                )
                st.session_state.dynamic_charts[i]["chart_type"] = new_type

            with col3:
                if chart_config.get("chart_type", "line") not in ["histogram", "heatmap"]:
                    x_options = (["timestamp"] + available_columns 
                               if "timestamp" in df.columns else available_columns)
                    if x_options:
                        current_x = chart_config.get("x_axis", x_options[0])
                        if current_x not in x_options:
                            current_x = x_options[0]
                        new_x = st.selectbox(
                            "X-Axis", options=x_options,
                            index=x_options.index(current_x) if current_x in x_options else 0,
                            key=f"x_{chart_config['id']}"
                        )
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
                        st.session_state.dynamic_charts[i]["y_axis"] = new_y

            with col5:
                if st.button("🗑️", key=f"delete_{chart_config['id']}"):
                    st.session_state.dynamic_charts.pop(i)
                    st.rerun()

            # Display chart
            if chart_config.get("chart_type") == "heatmap" or chart_config.get("y_axis"):
                fig = create_dynamic_chart(df, chart_config)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config['id']}")


def main():
    """Main dashboard function"""
    # Header
    st.markdown(
        '<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>',
        unsafe_allow_html=True,
    )

    initialize_session_state()

    # Initialize managers
    if not st.session_state.supabase_manager:
        st.session_state.supabase_manager = SupabaseManager()
        st.session_state.supabase_manager.connect()

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
                    st.session_state.subscriber = EnhancedTelemetrySubscriber()
                    if st.session_state.subscriber.connect():
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Failed!")
                st.rerun()

        with col2:
            if st.button("🛑 Disconnect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    st.session_state.subscriber = None
                st.info("🛑 Disconnected")
                st.rerun()

        # Load sessions
        if st.button("🔄 Refresh Sessions", use_container_width=True):
            if st.session_state.supabase_manager:
                st.session_state.available_sessions = st.session_state.supabase_manager.get_sessions()
                st.success("Sessions refreshed!")

        # Connection status
        stats = (st.session_state.subscriber.get_stats() 
                if st.session_state.subscriber else {
                    "messages_received": 0, "connection_attempts": 0, 
                    "errors": 0, "last_message_time": None, "last_error": None
                })

        render_connection_status(st.session_state.subscriber, st.session_state.supabase_manager, stats)

        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        if not st.session_state.is_historical_mode:
            new_auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
            if new_auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = new_auto_refresh

            if st.session_state.auto_refresh:
                refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)

    # Load available sessions if not loaded
    if not st.session_state.available_sessions and st.session_state.supabase_manager:
        st.session_state.available_sessions = st.session_state.supabase_manager.get_sessions()

    # Session selector
    render_session_selector()

    # Data loading logic
    df = pd.DataFrame()
    
    if st.session_state.data_source == "historical" and st.session_state.selected_session:
        # Historical mode - load data from selected session
        df = st.session_state.supabase_manager.get_session_data(st.session_state.selected_session)
        
    elif st.session_state.data_source == "realtime_session":
        # Real-time + Session mode - merge multiple data sources
        realtime_data = pd.DataFrame()
        history_data = pd.DataFrame()
        db_data = pd.DataFrame()
        
        # Get real-time data
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            new_messages = st.session_state.subscriber.get_messages()
            if new_messages:
                realtime_data = pd.DataFrame(new_messages)
                if 'timestamp' in realtime_data.columns:
                    realtime_data['timestamp'] = pd.to_datetime(realtime_data['timestamp'])

        # Get Ably history (async call would need to be handled differently in Streamlit)
        # For now, we'll focus on real-time and database data
        
        # Get current session data from database
        if st.session_state.supabase_manager and not realtime_data.empty:
            # Get the current session ID from real-time data
            current_session_id = realtime_data.iloc[0].get('session_id') if not realtime_data.empty else None
            if current_session_id:
                db_data = st.session_state.supabase_manager.get_current_session_data(current_session_id)

        # Merge all data sources
        df = merge_data_sources(realtime_data, history_data, db_data)
        
        # Update session state
        if not df.empty:
            st.session_state.telemetry_data = df
            st.session_state.last_update = datetime.now()

    # Display data status
    if df.empty:
        if st.session_state.data_source == "historical":
            st.warning("⏳ Please select a session to view historical data.")
        else:
            st.warning("⏳ Waiting for telemetry data... Make sure the bridge is running and connected.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Getting Started:**\n"
                "1. Ensure m1.py bridge is running\n"
                "2. Click 'Connect' to start receiving data\n"
                "3. Select your preferred data source"
            )
        return

    # Data summary
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points loaded")
    with col2:
        if not df.empty and 'timestamp' in df.columns:
            latest_time = df['timestamp'].max()
            st.info(f"⏰ Latest data: **{latest_time.strftime('%H:%M:%S')}**")
    with col3:
        if 'data_origin' in df.columns:
            origins = df['data_origin'].value_counts()
            st.success(f"📡 Sources: {len(origins)}")

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
        st.markdown("### 📊 Performance Overview")
        render_kpi_header(kpis)

    # Speed tab
    with tabs[1]:
        render_kpi_header(kpis)
        fig = create_speed_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Power tab
    with tabs[2]:
        render_kpi_header(kpis)
        fig = create_power_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # IMU tab
    with tabs[3]:
        render_kpi_header(kpis)
        fig = create_imu_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Efficiency tab
    with tabs[4]:
        render_kpi_header(kpis)
        fig = create_efficiency_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # GPS tab
    with tabs[5]:
        render_kpi_header(kpis)
        fig = create_gps_map(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Custom charts tab
    with tabs[6]:
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)

# Data tab (continued)
    with tabs[7]:
        render_kpi_header(kpis)
        st.subheader("📃 Raw Telemetry Data")
        
        # Data source information
        if 'data_origin' in df.columns:
            st.markdown("#### 📊 Data Source Breakdown")
            origin_counts = df['data_origin'].value_counts()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'realtime' in origin_counts:
                    st.metric("🔴 Real-time", f"{origin_counts['realtime']:,}")
            with col2:
                if 'ably_history' in origin_counts:
                    st.metric("📡 Ably History", f"{origin_counts['ably_history']:,}")
            with col3:
                if 'database' in origin_counts:
                    st.metric("💾 Database", f"{origin_counts['database']:,}")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning(
                "ℹ️ Only the **last 100 datapoints** are displayed below for performance. "
                "Download the CSV for the complete dataset."
            )
        with col2:
            show_all = st.checkbox("Show all data", help="⚠️ May be slow with large datasets")
        
        # Data display
        display_df = df if show_all else df.tail(100)
        
        # Add data origin color coding if available
        if 'data_origin' in display_df.columns:
            st.markdown("**Legend:** 🔴 Real-time | 📡 Ably History | 💾 Database")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete CSV",
                data=csv,
                file_name=f"telemetry_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col2:
            if not df.empty:
                # Create summary statistics
                summary_stats = {
                    'Total Records': len(df),
                    'Time Range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A',
                    'Session ID': df['session_id'].iloc[0] if 'session_id' in df.columns else 'N/A',
                    'Data Sources': ', '.join(df['data_origin'].unique()) if 'data_origin' in df.columns else 'N/A'
                }
                
                summary_json = json.dumps(summary_stats, indent=2, default=str)
                st.download_button(
                    label="📋 Download Summary",
                    data=summary_json,
                    file_name=f"telemetry_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # Auto-refresh logic (only for real-time mode)
    if (st.session_state.auto_refresh and 
        not st.session_state.is_historical_mode and
        st.session_state.subscriber and 
        st.session_state.subscriber.is_connected):
        
        time.sleep(refresh_interval)
        st.rerun()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>
            <p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | Enhanced with Historical Data & Session Management</p>
            <p>🚗 Real-time monitoring • 💾 Persistent storage • 📊 Advanced analytics • 🔄 Multi-source data integration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
