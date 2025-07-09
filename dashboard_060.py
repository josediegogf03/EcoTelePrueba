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
from typing import Dict, Any, List, Optional, Tuple
import threading
import queue
import asyncio
import uuid
import warnings

# Handles imports with error checking
try:
    from ably import AblyRealtime, AblyRest
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

# Disables tracemalloc warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Configuration
DASHBOARD_ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
DASHBOARD_CHANNEL_NAME = "telemetry-dashboard-channel"
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"

# Configures the Streamlit page
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Shell Eco-marathon Enhanced Telemetry Dashboard with Historical Data",
    },
)

# Enhanced CSS styling
st.markdown("""
<style>
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
        --shadow-color: rgba(0,0,0,0.1);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #fafafa;
            --text-secondary: #a0a0a0;
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --border-color: #4a4a4a;
            --shadow-color: rgba(255,255,255,0.1);
        }
    }

    .main-header {
        font-size: 2.2rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 0 2px 4px var(--shadow-color);
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
        box-shadow: 0 2px 8px var(--shadow-color);
    }

    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
    }

    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
    }

    .status-historical {
        background: linear-gradient(135deg, #e2e3e5 0%, #d1d3d4 100%);
        color: #495057;
        border: 2px solid #6c757d;
    }

    .data-source-card {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px var(--shadow-color);
    }

    .data-source-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow-color);
    }

    .session-info {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 16px var(--shadow-color);
    }

    .session-info h3 {
        color: var(--primary-color);
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }

    .session-info p {
        margin: 0.5rem 0;
        color: var(--text-primary);
        font-size: 0.95rem;
    }

    .historical-notice {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #ffc107;
        text-align: center;
        font-weight: 600;
    }

    .chart-type-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .chart-type-card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px var(--shadow-color);
    }

    .chart-type-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-color);
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
        line-height: 1.5;
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
        background: transparent; /* Changed for transparent background */
        border-bottom: 2px solid var(--border-color);
        border-radius: 8px 8px 0 0;
        padding: 0.5rem;
        /* box-shadow: 0 2px 8px var(--shadow-color); Removed or lightened shadow for transparent look */
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
        .chart-type-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Logger setup
def setup_terminal_logging():
    """Configures logging for terminal output."""
    logger = logging.getLogger("TelemetryDashboard")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

setup_terminal_logging()

class EnhancedTelemetryManager:
    """Enhanced telemetry manager with multi-source data integration."""
    
    def __init__(self):
        self.realtime_subscriber = None
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
            "data_sources": []
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self.logger = logging.getLogger("TelemetryDashboard")
        
    def connect_supabase(self) -> bool:
        """Connect to Supabase database."""
        try:
            self.supabase_client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
            self.logger.info("✅ Connected to Supabase")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Supabase: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return False
    
    def connect_realtime(self) -> bool:
        """Connect to Ably for real-time data."""
        try:
            with self._lock:
                self.stats["connection_attempts"] += 1
            
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
            self.logger.error(f"❌ Real-time connection failed: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return False
    
    def _connection_worker(self):
        """Worker thread for Ably connection."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_connection_handler())
        except Exception as e:
            self.logger.error(f"💥 Connection worker error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False
    
    async def _async_connection_handler(self):
        """Handle Ably connection asynchronously."""
        try:
            self.realtime_subscriber = AblyRealtime(DASHBOARD_ABLY_API_KEY)
            
            def on_connected(state_change):
                self.is_connected = True
                self.logger.info("✅ Connected to Ably")
            
            def on_disconnected(state_change):
                self.is_connected = False
                self.logger.warning("❌ Disconnected from Ably")
            
            def on_failed(state_change):
                self.is_connected = False
                self.logger.error(f"💥 Connection failed: {state_change}")
            
            self.realtime_subscriber.connection.on("connected", on_connected)
            self.realtime_subscriber.connection.on("disconnected", on_disconnected)
            self.realtime_subscriber.connection.on("failed", on_failed)
            
            await self.realtime_subscriber.connection.once_async("connected")
            
            channel = self.realtime_subscriber.channels.get(DASHBOARD_CHANNEL_NAME)
            await channel.subscribe("telemetry_update", self._on_message_received)
            
            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"💥 Async connection error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False
    
    def _on_message_received(self, message):
        """Handle incoming real-time messages."""
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
            
            with self._lock:
                # Limit queue size to avoid excessive memory usage in Streamlit
                if self.message_queue.qsize() > 500:
                    while self.message_queue.qsize() > 250: # Remove oldest messages
                        try:
                            self.message_queue.get_nowait()
                        except queue.Empty:
                            break
                
                # Add data source tag
                data['data_source'] = 'realtime'
                self.message_queue.put(data)
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = datetime.now()
                
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
    
    def get_realtime_messages(self) -> List[Dict[str, Any]]:
        """Get all queued real-time messages."""
        messages = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
        return messages
    
    def get_current_session_data(self, session_id: str) -> pd.DataFrame:
        """Get current session data from Supabase. Fetches all available records for the session."""
        try:
            if not self.supabase_client:
                return pd.DataFrame()
            
            # This will attempt to fetch all records for the given session ID.
            # Supabase Python client's execute() will handle pagination internally
            # if the dataset is very large, retrieving all results.
            response = self.supabase_client.table(SUPABASE_TABLE_NAME).select("*").eq("session_id", session_id).order("timestamp", desc=False).execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['data_source'] = 'supabase_current'
                return df
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching current session data from Supabase: {e}")
            return pd.DataFrame()
    
    def get_historical_sessions(self) -> List[Dict[str, Any]]:
        """Get list of historical sessions."""
        try:
            if not self.supabase_client:
                return []
            
            # Select distinct session_ids and their timestamps to determine start/end times
            # Note: For very large tables, this might be slow.
            # A dedicated 'sessions' table or view in Supabase would be more efficient.
            response = self.supabase_client.table(SUPABASE_TABLE_NAME).select("session_id, timestamp").order("timestamp", desc=True).execute()
            
            if not response.data:
                return []
            
            # Group by session_id and get session info
            sessions = {}
            for record in response.data:
                session_id = record['session_id']
                timestamp = record['timestamp']
                
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session_id': session_id,
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'record_count': 1
                    }
                else:
                    sessions[session_id]['record_count'] += 1
                    if timestamp < sessions[session_id]['start_time']:
                        sessions[session_id]['start_time'] = timestamp
                    if timestamp > sessions[session_id]['end_time']:
                        sessions[session_id]['end_time'] = timestamp
            
            # Convert to list and calculate duration
            session_list = []
            for session_info in sessions.values():
                try:
                    # Handle Z suffix for UTC
                    start_dt = datetime.fromisoformat(session_info['start_time'].replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(session_info['end_time'].replace('Z', '+00:00'))
                    duration = end_dt - start_dt
                    
                    session_list.append({
                        'session_id': session_info['session_id'],
                        'start_time': start_dt,
                        'end_time': end_dt,
                        'duration': duration,
                        'record_count': session_info['record_count']
                    })
                except Exception as e:
                    self.logger.error(f"Error processing session {session_info['session_id']}: {e}")
            
            return sorted(session_list, key=lambda x: x['start_time'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical sessions: {e}")
            return []
    
    def get_historical_data(self, session_id: str) -> pd.DataFrame:
        """Get historical data for a specific session. Fetches all available records for the session."""
        try:
            if not self.supabase_client:
                return pd.DataFrame()
            
            # This will attempt to fetch all records for the given session ID.
            response = self.supabase_client.table(SUPABASE_TABLE_NAME).select("*").eq("session_id", session_id).order("timestamp", desc=False).execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['data_source'] = 'supabase_historical'
                return df
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data from Supabase: {e}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Disconnect from all services."""
        try:
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False
            
            if self.realtime_subscriber:
                try:
                    self.realtime_subscriber.close()
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably: {e}")
            
            if self.connection_thread and self.connection_thread.is_alive():
                # Give a small timeout for the thread to finish cleanly
                self.connection_thread.join(timeout=5) 
            
            self.logger.info("🔚 Disconnected from services")
            
        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
        finally:
            self.realtime_subscriber = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        with self._lock:
            return self.stats.copy()

def merge_telemetry_data(realtime_data: List[Dict], 
                        supabase_data: pd.DataFrame, 
                        streamlit_history: pd.DataFrame) -> pd.DataFrame:
    """Merge data from multiple sources with deduplication."""
    try:
        all_data = []
        
        # Add real-time data
        if realtime_data:
            all_data.extend(realtime_data)
        
        # Add Supabase data
        if not supabase_data.empty:
            all_data.extend(supabase_data.to_dict('records'))
        
        # Add Streamlit history (data already in session_state)
        # We should prioritize fresh data sources, so merge from newest back to oldest.
        # Ensure we don't duplicate existing data if a Supabase fetch brings older history.
        
        # If there's existing data, treat it as part of the base for merging
        if not streamlit_history.empty:
            all_data.extend(streamlit_history.to_dict('records'))
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            # Coerce errors means invalid dates become NaT (Not a Time)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        else:
            # If no timestamp, can't sort or dedup effectively, return what we have
            return df
        
        # Remove duplicates based on timestamp and a unique identifier like message_id if available.
        # Keep the latest entry in case of timestamp overlap (e.g., real-time vs. Supabase for same point)
        dedup_columns = ['timestamp']
        if 'message_id' in df.columns:
            dedup_columns.append('message_id')
        
        # Drop duplicates, keeping the 'last' (most recently added in `all_data` list,
        # which typically means the real-time or newest fetch takes precedence)
        df = df.drop_duplicates(subset=dedup_columns, keep='last')
        
        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error merging telemetry data: {e}")
        return pd.DataFrame()

def initialize_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "telemetry_manager": None,
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "data_source_mode": "realtime_session",  # or "historical"
        "selected_session": None,
        "historical_sessions": [],
        "current_session_id": None,
        "is_viewing_historical": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate KPIs from telemetry data."""
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
            "voltage_v", "current_a", # Added for power chart calculations
            "accel_x", "accel_y", "accel_z", # Added for IMU chart calculations
            "latitude", "longitude" # Added for GPS map calculations
        ]
        
        # Convert relevant columns to numeric, coercing errors to NaN
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        kpis = default_kpis.copy()
        
        # Ensure latest value is used for cumulative metrics
        if "energy_j" in df.columns and not df["energy_j"].dropna().empty:
            kpis["total_energy_mj"] = max(0, df["energy_j"].dropna().iloc[-1] / 1_000_000)
        
        if "speed_ms" in df.columns:
            speed_data = df["speed_ms"].dropna()
            if not speed_data.empty:
                kpis["max_speed_ms"] = max(0, speed_data.max())
                kpis["avg_speed_ms"] = max(0, speed_data.mean())
        
        if "distance_m" in df.columns and not df["distance_m"].dropna().empty:
            kpis["total_distance_km"] = max(0, df["distance_m"].dropna().iloc[-1] / 1000)
        
        if "power_w" in df.columns:
            power_data = df["power_w"].dropna()
            if not power_data.empty:
                kpis["avg_power_w"] = max(0, power_data.mean())
        
        if kpis["total_energy_mj"] > 0:
            kpis["efficiency_km_per_mj"] = kpis["total_distance_km"] / kpis["total_energy_mj"]
        
        # Calculate total acceleration if components are present
        if all(col in df.columns for col in ["accel_x", "accel_y", "accel_z"]):
            # Recalculate total_acceleration if not directly provided
            accel_x = df["accel_x"].dropna()
            accel_y = df["accel_y"].dropna()
            accel_z = df["accel_z"].dropna()
            if not accel_x.empty and not accel_y.empty and not accel_z.empty:
                total_acceleration_calculated = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                if not total_acceleration_calculated.empty:
                    kpis["max_acceleration"] = max(0, total_acceleration_calculated.max())
            elif "total_acceleration" in df.columns: # Fallback to existing if calculated isn't possible
                accel_data = df["total_acceleration"].dropna()
                if not accel_data.empty:
                    kpis["max_acceleration"] = max(0, accel_data.max())
        elif "total_acceleration" in df.columns:
            accel_data = df["total_acceleration"].dropna()
            if not accel_data.empty:
                kpis["max_acceleration"] = max(0, accel_data.max())


        if all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z"]):
            gyro_data_df = df[["gyro_x", "gyro_y", "gyro_z"]].dropna()
            if not gyro_data_df.empty:
                gyro_magnitude = np.sqrt(
                    gyro_data_df["gyro_x"] ** 2 + gyro_data_df["gyro_y"] ** 2 + gyro_data_df["gyro_z"] ** 2
                )
                if not gyro_magnitude.empty:
                    kpis["avg_gyro_magnitude"] = max(0, gyro_magnitude.mean())
        
        return kpis
        
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis

def render_kpi_header(kpis: Dict[str, float]):
    """Render KPI header with metrics."""
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
    """Render overview tab with enhanced KPIs."""
    st.markdown("### 📊 Performance Overview")
    st.markdown("Real-time key performance indicators for your Shell Eco-marathon vehicle")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🛣️ Total Distance",
            value=f"{kpis['total_distance_km']:.2f} km",
            help="Distance traveled during the session"
        )
        st.metric(
            label="🔋 Energy Consumed",
            value=f"{kpis['total_energy_mj']:.2f} MJ",
            help="Total energy consumption"
        )
    
    with col2:
        st.metric(
            label="🚀 Maximum Speed",
            value=f"{kpis['max_speed_ms']:.1f} m/s",
            help="Highest speed achieved"
        )
        st.metric(
            label="💡 Average Power",
            value=f"{kpis['avg_power_w']:.1f} W",
            help="Mean power consumption"
        )
    
    with col3:
        st.metric(
            label="🏃 Average Speed",
            value=f"{kpis['avg_speed_ms']:.1f} m/s",
            help="Mean speed throughout the session"
        )
        st.metric(
            label="♻️ Efficiency",
            value=f"{kpis['efficiency_km_per_mj']:.2f} km/MJ",
            help="Energy efficiency ratio"
        )
    
    with col4:
        st.metric(
            label="📈 Max Acceleration",
            value=f"{kpis['max_acceleration']:.2f} m/s²",
            help="Peak acceleration recorded"
        )
        st.metric(
            label="🎯 Avg Gyro Magnitude",
            value=f"{kpis['avg_gyro_magnitude']:.2f} °/s",
            help="Average rotational movement"
        )

def render_session_info(session_data: Dict[str, Any]):
    """Render session information card."""
    st.markdown(f"""
    <div class="session-info">
        <h3>📊 Session Information</h3>
        <p>📋 <strong>Session:</strong> {session_data['session_id'][:8]}...</p>
        <p>📅 <strong>Start:</strong> {session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>⏱️ <strong>Duration:</strong> {session_data['duration']}</p>
        <p>📊 <strong>Records:</strong> {session_data['record_count']:,}</p>
    </div>
    """, unsafe_allow_html=True)

def create_speed_chart(df: pd.DataFrame):
    """Create speed chart."""
    if df.empty or "speed_ms" not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    fig = px.line(
        df, x="timestamp", y="speed_ms",
        title="🚗 Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4"],
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    
    return fig

def create_power_chart(df: pd.DataFrame):
    """Create power chart."""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
        vertical_spacing=0.15,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["voltage_v"],
            name="Voltage (V)", line=dict(color="#2ca02c", width=2),
        ), row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["current_a"],
            name="Current (A)", line=dict(color="#d62728", width=2),
        ), row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["power_w"],
            name="Power (W)", line=dict(color="#ff7f0e", width=2),
        ), row=2, col=1,
    )
    
    fig.update_layout(
        height=500,
        title_text="⚡ Electrical System Performance",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_imu_chart(df: pd.DataFrame):
    """Create IMU chart."""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
        vertical_spacing=0.25,
    )
    
    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[axis],
                name=f"Gyro {axis[-1].upper()}",
                line=dict(color=colors_gyro[i], width=2),
            ), row=1, col=1,
        )
    
    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[axis],
                name=f"Accel {axis[-1].upper()}",
                line=dict(color=colors_accel[i], width=2),
            ), row=2, col=1,
        )
    
    fig.update_layout(
        height=600,
        title_text="🎮 IMU Sensor Data Analysis",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_imu_detail_chart(df: pd.DataFrame):
    """Create detailed IMU chart."""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("🌀 Gyro X", "🌀 Gyro Y", "🌀 Gyro Z", "📊 Accel X", "📊 Accel Y", "📊 Accel Z"),
        vertical_spacing=0.3,
        horizontal_spacing=0.1,
    )
    
    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]
    
    for i, (axis, color) in enumerate(zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[axis],
                name=f"Gyro {axis[-1].upper()}",
                line=dict(color=color, width=2),
                showlegend=False,
            ), row=1, col=i + 1,
        )
    
    for i, (axis, color) in enumerate(zip(["accel_x", "accel_y", "accel_z"], accel_colors)):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[axis],
                name=f"Accel {axis[-1].upper()}",
                line=dict(color=color, width=2),
                showlegend=False,
            ), row=2, col=i + 1,
        )
    
    fig.update_layout(
        height=600,
        title_text="🎮 Detailed IMU Sensor Analysis",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_efficiency_chart(df: pd.DataFrame):
    """Create efficiency chart."""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    fig = px.scatter(
        df, x="speed_ms", y="power_w",
        color="voltage_v" if "voltage_v" in df.columns else None,
        title="⚡ Efficiency Analysis: Speed vs Power Consumption",
        labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
        color_continuous_scale="viridis",
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    
    return fig

def create_gps_map(df: pd.DataFrame):
    """Create GPS map."""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    df_valid = df.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    center_point = dict(
        lat=df_valid["latitude"].mean(),
        lon=df_valid["longitude"].mean()
    )
    
    fig = px.scatter_map(
        df_valid,
        lat="latitude", lon="longitude",
        color="speed_ms" if "speed_ms" in df_valid.columns else None,
        size="power_w" if "power_w" in df_valid.columns else None,
        hover_data=["speed_ms", "power_w", "voltage_v"] if all(col in df_valid.columns for col in ["speed_ms", "power_w", "voltage_v"]) else None,
        map_style="open-street-map",
        title="🛰️ Vehicle Track and Performance",
        height=400,
        zoom=15,
        center=center_point,
        color_continuous_scale="plasma",
    )
    
    return fig

def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get available numeric columns for plotting."""
    if df.empty:
        return []
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"] # Exclude internal/meta columns
    
    # Also exclude lat/lon from general charts unless they are specific to GPS map
    # They are useful for dynamic charts IF the user explicitly chooses them
    return [col for col in numeric_columns if col not in exclude_cols]

def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create dynamic chart based on configuration."""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")
    
    # Handle heatmap case where x_col and y_col are not applicable in the same way
    if chart_type == "heatmap":
        numeric_cols = get_available_columns(df)
        if len(numeric_cols) >= 2:
            # Ensure all columns in corr_matrix are numeric and drop NaNs for correlation
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="🔥 Correlation Heatmap", color_continuous_scale="RdBu_r", aspect="auto")
        else:
            return go.Figure().add_annotation(
                text="Need at least 2 numeric columns for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
    else: # For other chart types, x_col and y_col are crucial
        if not y_col or y_col not in df.columns:
            return go.Figure().add_annotation(
                text="Invalid Y-axis column selection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
        
        if x_col not in df.columns:
             return go.Figure().add_annotation(
                text="Invalid X-axis column selection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )

        try:
            if chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#ff7f0e"])
            elif chart_type == "bar":
                # For bar, it often makes sense to show recent data or aggregates.
                # Here, showing the last 20 points, but this can be adjusted.
                recent_df = df.tail(20) 
                if recent_df.empty:
                    return go.Figure().add_annotation(
                        text="Not enough recent data for bar chart",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                    )
                fig = px.bar(recent_df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#2ca02c"])
            elif chart_type == "histogram":
                # Histogram uses only one axis (the value itself) for distribution
                fig = px.histogram(df, x=y_col, title=f"Distribution of {y_col}", color_discrete_sequence=["#d62728"])
            else: # Fallback to line
                fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
        
    fig.update_layout(
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def render_dynamic_charts_section(df: pd.DataFrame):
    """Render dynamic charts section."""
    st.markdown("""
    <div class="data-source-card">
        <h3>🎯 Create Custom Charts</h3>
        <p>Click <strong>"Add Chart"</strong> to create custom visualizations with your preferred variables and chart types.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
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
    """, unsafe_allow_html=True)
    
    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []
    
    if not available_columns:
        st.warning("⏳ No numeric data available for creating charts. Connect and wait for data.")
        return
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart"):
            try:
                new_chart = {
                    "id": str(uuid.uuid4()),
                    "title": "New Chart",
                    "chart_type": "line",
                    "x_axis": "timestamp" if "timestamp" in df.columns else (available_columns[0] if available_columns else None),
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
        # Loop through a copy to allow modification during iteration (e.g., pop)
        for i, chart_config in enumerate(list(st.session_state.dynamic_charts)): 
            try:
                with st.container(border=True):
                    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])
                    
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
                            options=["line", "scatter", "bar", "histogram", "heatmap"],
                            index=["line", "scatter", "bar", "histogram", "heatmap"].index(chart_config.get("chart_type", "line")),
                            key=f"type_{chart_config['id']}",
                        )
                        if new_type != chart_config.get("chart_type"):
                            st.session_state.dynamic_charts[i]["chart_type"] = new_type
                    
                    with col3:
                        # X-axis only for non-histogram/heatmap
                        if chart_config.get("chart_type", "line") not in ["histogram", "heatmap"]:
                            x_options = ["timestamp"] + available_columns if "timestamp" in df.columns else available_columns
                            current_x = chart_config.get("x_axis")
                            if current_x not in x_options and x_options:
                                current_x = x_options[0] # Fallback if current_x is invalid
                            
                            if x_options:
                                new_x = st.selectbox(
                                    "X-Axis",
                                    options=x_options,
                                    index=x_options.index(current_x) if current_x in x_options else 0,
                                    key=f"x_{chart_config['id']}",
                                )
                                if new_x != chart_config.get("x_axis"):
                                    st.session_state.dynamic_charts[i]["x_axis"] = new_x
                            else:
                                st.empty() # Placeholder if no options
                    
                    with col4:
                        # Y-axis only for non-heatmap
                        if chart_config.get("chart_type", "line") != "heatmap":
                            if available_columns:
                                current_y = chart_config.get("y_axis")
                                if current_y not in available_columns:
                                    current_y = available_columns[0] # Fallback if current_y is invalid
                                
                                new_y = st.selectbox(
                                    "Y-Axis",
                                    options=available_columns,
                                    index=available_columns.index(current_y) if current_y in available_columns else 0,
                                    key=f"y_{chart_config['id']}",
                                )
                                if new_y != chart_config.get("y_axis"):
                                    st.session_state.dynamic_charts[i]["y_axis"] = new_y
                            else:
                                st.empty() # Placeholder if no options
                    
                    with col5:
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"):
                            try:
                                # Find index of current chart_config using its unique ID
                                idx_to_delete = next((j for j, cfg in enumerate(st.session_state.dynamic_charts) if cfg['id'] == chart_config['id']), -1)
                                if idx_to_delete != -1:
                                    st.session_state.dynamic_charts.pop(idx_to_delete)
                                st.rerun() # Rerun to reflect deletion
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")
                    
                    try:
                        # Only try to create chart if necessary columns are selected based on chart type
                        if chart_config.get("chart_type") == "heatmap":
                            fig = create_dynamic_chart(df, chart_config)
                        elif chart_config.get("y_axis") and \
                             (chart_config.get("chart_type") == "histogram" or chart_config.get("x_axis")):
                             fig = create_dynamic_chart(df, chart_config)
                        else:
                            fig = None # Don't render if selections are incomplete for current type
                            st.warning("Please select valid axes for the chosen chart type.")

                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_plot_{chart_config['id']}")
                        
                    except Exception as e:
                        st.error(f"Error rendering chart: {e}")
            
            except Exception as e:
                st.error(f"Error rendering chart configuration: {e}")

def main():
    """Main dashboard function."""
    st.markdown('<div class="main-header">🏎️ Shell Eco-marathon Enhanced Telemetry Dashboard</div>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar for connection and data source selection
    with st.sidebar:
        st.header("🔧 Connection & Data Source")
        
        # Data source selection
        data_source_mode = st.radio(
            "📊 Data Source",
            options=["realtime_session", "historical"],
            format_func=lambda x: "🔴 Real-time + Session Data" if x == "realtime_session" else "📚 Historical Data",
            key="data_source_mode_radio"
        )
        
        # If data source mode changes, reset telemetry data and flags
        if data_source_mode != st.session_state.data_source_mode:
            st.session_state.data_source_mode = data_source_mode
            st.session_state.telemetry_data = pd.DataFrame()
            st.session_state.is_viewing_historical = (data_source_mode == "historical")
            st.session_state.selected_session = None
            st.session_state.current_session_id = None
            st.rerun()
        
        # Real-time mode controls
        if st.session_state.data_source_mode == "realtime_session":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect", use_container_width=True):
                    if st.session_state.telemetry_manager:
                        st.session_state.telemetry_manager.disconnect()
                        time.sleep(0.5) # Give a moment for cleanup
                    
                    with st.spinner("Connecting..."):
                        st.session_state.telemetry_manager = EnhancedTelemetryManager()
                        
                        # Connect to Supabase
                        supabase_connected = st.session_state.telemetry_manager.connect_supabase()
                        
                        # Connect to real-time
                        realtime_connected = False
                        if ABLY_AVAILABLE: # Only try if Ably is installed
                            realtime_connected = st.session_state.telemetry_manager.connect_realtime()
                        
                        if supabase_connected and realtime_connected:
                            st.success("✅ Connected!")
                        elif supabase_connected:
                            st.warning("⚠️ Supabase only connected (Ably not available or failed)")
                        else:
                            st.error("❌ Failed to connect to any service!")
                    
                    st.rerun()
            
            with col2:
                if st.button("🛑 Disconnect", use_container_width=True):
                    if st.session_state.telemetry_manager:
                        st.session_state.telemetry_manager.disconnect()
                        st.session_state.telemetry_manager = None
                    st.info("🛑 Disconnected")
                    st.rerun()
            
            # Connection status display for real-time mode
            if st.session_state.telemetry_manager:
                stats = st.session_state.telemetry_manager.get_stats()
                
                if st.session_state.telemetry_manager.is_connected:
                    st.markdown('<div class="status-indicator status-connected">✅ Real-time Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-indicator status-disconnected">❌ Real-time Disconnected</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
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
                
                if stats["last_error"]:
                    st.error(f"⚠️ {stats['last_error'][:40]}...") # Show truncated error
            
            st.divider()
            
            # Auto-refresh settings
            st.subheader("⚙️ Settings")
            new_auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh, 
                                            help="Automatically refresh data from real-time stream")
            
            if new_auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = new_auto_refresh
            
            refresh_interval = 3 # Default refresh interval
            if st.session_state.auto_refresh:
                refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)
            
            # Store refresh interval for use in main loop
            st.session_state.refresh_interval = refresh_interval
            
        else: # Historical data mode controls
            st.markdown('<div class="status-indicator status-historical">📚 Historical Mode</div>', unsafe_allow_html=True)
            
            if not st.session_state.telemetry_manager:
                st.session_state.telemetry_manager = EnhancedTelemetryManager()
                st.session_state.telemetry_manager.connect_supabase()
            
            if st.button("🔄 Refresh Sessions", use_container_width=True):
                with st.spinner("Loading sessions..."):
                    st.session_state.historical_sessions = st.session_state.telemetry_manager.get_historical_sessions()
                st.rerun()
            
            # Session selection
            if st.session_state.historical_sessions:
                session_options = []
                for session in st.session_state.historical_sessions:
                    session_options.append(f"{session['session_id'][:8]}... - {session['start_time'].strftime('%Y-%m-%d %H:%M')} ({session['record_count']:,} records)")
                
                selected_session_idx = st.selectbox(
                    "📋 Select Session",
                    options=range(len(session_options)),
                    format_func=lambda x: session_options[x],
                    key="session_selector",
                    index=0 # Default to the most recent session
                )
                
                if selected_session_idx is not None:
                    selected_session = st.session_state.historical_sessions[selected_session_idx]
                    
                    # Only load if a different session is selected or if data is empty
                    if st.session_state.selected_session is None or \
                       st.session_state.selected_session['session_id'] != selected_session['session_id'] or \
                       st.session_state.telemetry_data.empty:
                        st.session_state.selected_session = selected_session
                        st.session_state.is_viewing_historical = True
                        
                        # Load historical data
                        with st.spinner(f"Loading data for session {selected_session['session_id'][:8]}..."):
                            historical_df = st.session_state.telemetry_manager.get_historical_data(selected_session['session_id'])
                            st.session_state.telemetry_data = historical_df
                            st.session_state.last_update = datetime.now()
                        st.rerun() # Rerun to display loaded data
            else:
                st.info("Click 'Refresh Sessions' to load available sessions from Supabase.")
        
        st.info(f"📡 Channel: {DASHBOARD_CHANNEL_NAME}")
    
    # Main content area
    df = st.session_state.telemetry_data.copy()
    new_messages_count = 0
    
    # Data ingestion logic
    if st.session_state.data_source_mode == "realtime_session":
        # Get real-time messages and current session data
        if st.session_state.telemetry_manager and st.session_state.telemetry_manager.is_connected:
            new_messages = st.session_state.telemetry_manager.get_realtime_messages()
            
            current_session_data_from_supabase = pd.DataFrame()
            # If new messages are arriving, update current_session_id and fetch its historical part
            if new_messages and 'session_id' in new_messages[0]:
                current_session_id = new_messages[0]['session_id']
                if st.session_state.current_session_id != current_session_id or \
                   st.session_state.telemetry_data.empty: # Load full session initially or if session changes
                    st.session_state.current_session_id = current_session_id
                    current_session_data_from_supabase = st.session_state.telemetry_manager.get_current_session_data(current_session_id)
                
            # Merge new real-time data with existing and (optionally) full current session data
            if new_messages or not current_session_data_from_supabase.empty:
                merged_data = merge_telemetry_data(
                    new_messages,
                    current_session_data_from_supabase,
                    st.session_state.telemetry_data
                )
                
                if not merged_data.empty:
                    new_messages_count = len(new_messages) if new_messages else 0
                    st.session_state.telemetry_data = merged_data
                    st.session_state.last_update = datetime.now()
        
        st.session_state.is_viewing_historical = False # Ensure this is false in real-time mode
    
    elif st.session_state.data_source_mode == "historical":
        # Data is already loaded via button click in sidebar, no auto-refresh or new messages in this mode
        st.session_state.is_viewing_historical = True
        
    df = st.session_state.telemetry_data.copy() # Get the latest data for display

    # Show historical notice if viewing historical data
    if st.session_state.is_viewing_historical and st.session_state.selected_session:
        st.markdown(
            '<div class="historical-notice">📚 Viewing Historical Data - No auto-refresh active</div>',
            unsafe_allow_html=True
        )
        render_session_info(st.session_state.selected_session)
    
    # Empty state message
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.data_source_mode == "realtime_session":
                st.info(
                    "**Getting Started (Real-time):**\n"
                    "1. Ensure the bridge (your data sending script) is running\n"
                    "2. Click 'Connect' in the sidebar to start receiving data"
                )
            else:
                st.info(
                    "**Getting Started (Historical):**\n"
                    "1. Click 'Refresh Sessions' in the sidebar to load available sessions\n"
                    "2. Select a session and its data will load automatically"
                )
        
        with col2:
            with st.expander("🔍 Debug Information"):
                debug_info = {
                    "Data Source Mode": st.session_state.data_source_mode,
                    "Is Viewing Historical": st.session_state.is_viewing_historical,
                    "Selected Session ID": st.session_state.selected_session['session_id'][:8] + "..." if st.session_state.selected_session else None,
                    "Current Real-time Session ID": st.session_state.current_session_id,
                    "Number of Historical Sessions": len(st.session_state.historical_sessions),
                    "Telemetry Data Points (in memory)": len(st.session_state.telemetry_data),
                }
                
                if st.session_state.telemetry_manager:
                    stats = st.session_state.telemetry_manager.get_stats()
                    debug_info.update({
                        "Ably Connected (Manager Status)": st.session_state.telemetry_manager.is_connected,
                        "Messages Received (via Ably)": stats["messages_received"],
                        "Connection Errors": stats["errors"],
                    })
                
                st.json(debug_info)
        return
    
    # Status row for populated data
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points available")
    with col2:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col3:
        if st.session_state.data_source_mode == "realtime_session" and new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Tabs for different visualizations
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
    
    # Render content for each tab
    with tabs[0]:
        render_overview_tab(kpis)
    
    with tabs[1]:
        render_kpi_header(kpis)
        fig = create_speed_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        render_kpi_header(kpis)
        fig = create_power_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        render_kpi_header(kpis)
        fig = create_imu_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        render_kpi_header(kpis)
        fig = create_imu_detail_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        render_kpi_header(kpis)
        fig = create_efficiency_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[6]:
        render_kpi_header(kpis)
        fig = create_gps_map(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[7]:
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)
    
    with tabs[8]:
        render_kpi_header(kpis)
        
        st.subheader("📃 Raw Telemetry Data")
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
    
    # Auto-refresh for real-time mode only
    if (st.session_state.data_source_mode == "realtime_session" and 
        st.session_state.auto_refresh and 
        st.session_state.telemetry_manager and 
        st.session_state.telemetry_manager.is_connected):
        
        time.sleep(st.session_state.get('refresh_interval', 3)) # Use configured interval
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>"
        "<p><strong>Shell Eco-marathon Enhanced Telemetry Dashboard</strong> | Multi-Source Data Integration</p>"
        "<p>🚗 Real-time + Session Data | 📚 Historical Analysis | 🔄 Data Triangulation</p>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
