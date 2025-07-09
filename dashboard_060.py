# dashboard_enhanced.py - Enhanced Telemetry Dashboard with Supabase Integration
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
from typing import Dict, Any, List, Optional, Union
import threading
import queue
import asyncio
import uuid
import hashlib

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
    """Configures the dashboard logger to print to the terminal."""
    logger = logging.getLogger("TelemetryDashboard")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# Initializes terminal logging at application start.
setup_terminal_logging()

# Configuration constants
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"
MAX_DATAPOINTS = 1000000000  # 1 billion data points limit

# Configures the Streamlit page
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ChosF/EcoTele",
        "Report a bug": "https://github.com/ChosF/EcoTele/issues",
        "About": "Shell Eco-marathon Real-time Telemetry Dashboard v2.0",
    },
)

# Enhanced CSS with improved theming and modern styling
st.markdown("""
<style>
    /* Theme-aware color variables */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --error-color: #d62728;
        --text-primary: #262730;
        --text-secondary: #6c757d;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --border-color: #dee2e6;
        --accent-color: #17a2b8;
        --hover-color: #e9ecef;
    }

    /* Dark theme overrides */
    [data-theme="dark"] {
        --text-primary: #fafafa;
        --text-secondary: #a0a0a0;
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --border-color: #4a4a4a;
        --hover-color: #2a2a2a;
    }

    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }

    .status-indicator::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.6s;
        opacity: 0;
    }

    .status-indicator:hover::before {
        animation: shine 1.5s infinite;
    }

    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); opacity: 0; }
    }

    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border-color: #28a745;
        box-shadow: 0 4px 16px rgba(40, 167, 69, 0.2);
    }

    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border-color: #dc3545;
        box-shadow: 0 4px 16px rgba(220, 53, 69, 0.2);
    }

    .status-connecting {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border-color: #ffc107;
        box-shadow: 0 4px 16px rgba(255, 193, 7, 0.2);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }

    .data-source-selector {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .data-source-selector:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .session-info-card {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--primary-color) 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }

    .session-info-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transform: rotate(45deg);
    }

    .session-info-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        z-index: 1;
        position: relative;
    }

    .session-info-item strong {
        margin-left: 0.5rem;
        font-weight: 600;
    }

    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--bg-primary);
        padding: 1rem 0;
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .metric-card {
        background: var(--bg-primary);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(31, 119, 180, 0.1);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .metric-card:hover::before {
        transform: scaleX(1);
    }

    .stButton > button {
        border-radius: 12px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.3s ease, height 0.3s ease;
    }

    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }

    .stButton > button:hover {
        background: transparent;
        color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(31, 119, 180, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 50;
        background: var(--bg-primary);
        border-bottom: 2px solid var(--border-color);
        border-radius: 12px 12px 0 0;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .chart-container {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .chart-container:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(31, 119, 180, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .session-info-card {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 0.75rem;
        }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
        transition: background 0.3s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
</style>
""", unsafe_allow_html=True)


class DataTriangulator:
    """Handles data triangulation to prevent duplicates and ensure reliability."""
    
    def __init__(self):
        self.seen_hashes = set()
        self.logger = logging.getLogger("DataTriangulator")
    
    def create_data_hash(self, data: Dict[str, Any]) -> str:
        """Creates a unique hash for a data point based on key fields."""
        # Use timestamp and critical fields for hash generation
        key_fields = ['timestamp', 'session_id', 'speed_ms', 'voltage_v', 'current_a']
        hash_data = {}
        
        for field in key_fields:
            if field in data:
                hash_data[field] = data[field]
        
        # Create hash from sorted key-value pairs
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def is_duplicate(self, data: Dict[str, Any]) -> bool:
        """Checks if a data point is a duplicate."""
        data_hash = self.create_data_hash(data)
        
        if data_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(data_hash)
        
        # Clean up old hashes to prevent memory issues
        if len(self.seen_hashes) > 100000:  # Keep last 100k hashes
            # Remove oldest half
            hashes_to_remove = list(self.seen_hashes)[:50000]
            for hash_to_remove in hashes_to_remove:
                self.seen_hashes.discard(hash_to_remove)
        
        return False
    
    def triangulate_data_sources(self, realtime_data: List[Dict], 
                                 supabase_data: List[Dict], 
                                 history_data: List[Dict]) -> List[Dict]:
        """Triangulates data from multiple sources to create a unified dataset."""
        all_data = []
        
        # Combine all data sources
        combined_data = realtime_data + supabase_data + history_data
        
        # Sort by timestamp to maintain chronological order
        combined_data.sort(key=lambda x: x.get('timestamp', ''))
        
        # Filter duplicates
        for data_point in combined_data:
            if not self.is_duplicate(data_point):
                all_data.append(data_point)
        
        self.logger.info(f"Triangulated {len(all_data)} unique data points from {len(combined_data)} total points")
        return all_data


class SupabaseManager:
    """Manages Supabase database operations."""
    
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
        self.logger = logging.getLogger("SupabaseManager")
    
    def get_current_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all data for the current session from Supabase."""
        try:
            response = (
                self.client
                .table(SUPABASE_TABLE_NAME)
                .select("*")
                .eq("session_id", session_id)
                .order("timestamp", desc=False)
                .execute()
            )
            
            if response.data:
                self.logger.info(f"Retrieved {len(response.data)} records for session {session_id[:8]}...")
                return response.data
            else:
                self.logger.info(f"No data found for session {session_id}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving current session data: {e}")
            return []
    
    def get_available_sessions(self) -> List[Dict[str, Any]]:
        """Retrieves all available sessions with metadata."""
        try:
            # Get unique sessions with basic info
            response = (
                self.client
                .table(SUPABASE_TABLE_NAME)
                .select("session_id, timestamp")
                .order("timestamp", desc=True)
                .execute()
            )
            
            if not response.data:
                return []
            
            # Group by session and calculate metadata
            sessions = {}
            for record in response.data:
                session_id = record['session_id']
                timestamp = record['timestamp']
                
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session_id': session_id,
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'record_count': 0
                    }
                
                sessions[session_id]['record_count'] += 1
                
                # Update start and end times
                if timestamp < sessions[session_id]['start_time']:
                    sessions[session_id]['start_time'] = timestamp
                if timestamp > sessions[session_id]['end_time']:
                    sessions[session_id]['end_time'] = timestamp
            
            # Calculate durations
            for session_info in sessions.values():
                try:
                    start_dt = datetime.fromisoformat(session_info['start_time'].replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(session_info['end_time'].replace('Z', '+00:00'))
                    session_info['duration'] = end_dt - start_dt
                except Exception as e:
                    self.logger.warning(f"Error calculating duration for session {session_info['session_id']}: {e}")
                    session_info['duration'] = timedelta(0)
            
            # Sort by start time (most recent first)
            session_list = list(sessions.values())
            session_list.sort(key=lambda x: x['start_time'], reverse=True)
            
            self.logger.info(f"Found {len(session_list)} available sessions")
            return session_list
            
        except Exception as e:
            self.logger.error(f"Error retrieving available sessions: {e}")
            return []
    
    def get_historical_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all data for a specific historical session."""
        try:
            response = (
                self.client
                .table(SUPABASE_TABLE_NAME)
                .select("*")
                .eq("session_id", session_id)
                .order("timestamp", desc=False)
                .execute()
            )
            
            if response.data:
                self.logger.info(f"Retrieved {len(response.data)} records for historical session {session_id[:8]}...")
                return response.data
            else:
                self.logger.info(f"No data found for historical session {session_id}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving historical session data: {e}")
            return []


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

            time.sleep(2)  # Allow connection to establish
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
        """Handle Ably connection asynchronously"""
        try:
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

            await self.ably_client.connection.once_async("connected")

            self.channel = self.ably_client.channels.get(CHANNEL_NAME)
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

            with self._lock:
                if self.message_queue.qsize() > 1000:  # Prevent memory issues
                    while self.message_queue.qsize() > 500:
                        try:
                            self.message_queue.get_nowait()
                        except queue.Empty:
                            break

                self.message_queue.put(data)
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = datetime.now()

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
        return messages

    def disconnect(self):
        """Disconnect from Ably"""
        try:
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False

            if self.ably_client:
                try:
                    self.ably_client.close()
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably: {e}")

            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=5)

        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
        finally:
            self.ably_client = None
            self.channel = None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self._lock:
            return self.stats.copy()


def initialize_session_state():
    """Initializes Streamlit session state variables."""
    defaults = {
        "subscriber": None,
        "supabase_manager": SupabaseManager(),
        "data_triangulator": DataTriangulator(),
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "data_source": "realtime_recent",  # "realtime_recent" or "historical"
        "current_session_id": None,
        "selected_historical_session": None,
        "available_sessions": [],
        "session_info": None,
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


def render_session_info(session_info: Dict[str, Any]):
    """Renders session information in a beautiful card format."""
    if not session_info:
        return
    
    # Format the session info display
    session_id_short = session_info['session_id'][:8] + "..." if len(session_info['session_id']) > 8 else session_info['session_id']
    
    try:
        start_time = datetime.fromisoformat(session_info['start_time'].replace('Z', '+00:00'))
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    except:
        start_time_str = session_info.get('start_time', 'Unknown')
    
    duration = session_info.get('duration', timedelta(0))
    duration_str = str(duration).split('.')[0]  # Remove microseconds
    
    record_count = session_info.get('record_count', 0)
    
    # Display session info card
    st.markdown(f"""
    <div class="session-info-card">
        <div class="session-info-item">
            📋 <strong>Session: {session_id_short}</strong>
        </div>
        <div class="session-info-item">
            📅 <strong>Start: {start_time_str}</strong>
        </div>
        <div class="session-info-item">
            ⏱️ <strong>Duration: {duration_str}</strong>
        </div>
        <div class="session-info-item">
            📊 <strong>Records: {record_count:,}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_header(kpis: Dict[str, float]):
    """Renders KPI header with enhanced styling."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📏 Distance", f"{kpis['total_distance_km']:.2f} km")
        st.metric("🔋 Energy", f"{kpis['total_energy_mj']:.2f} MJ")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🚀 Max Speed", f"{kpis['max_speed_ms']:.1f} m/s")
        st.metric("💡 Avg Power", f"{kpis['avg_power_w']:.1f} W")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏃 Avg Speed", f"{kpis['avg_speed_ms']:.1f} m/s")
        st.metric("♻️ Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Max Acc.", f"{kpis['max_acceleration']:.2f} m/s²")
        st.metric("🎯 Avg Gyro", f"{kpis['avg_gyro_magnitude']:.2f} °/s")
        st.markdown('</div>', unsafe_allow_html=True)


def render_data_source_selector():
    """Renders the data source selection interface."""
    st.markdown('<div class="data-source-selector">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Source Selection")
    
    # Data source selection
    data_source = st.radio(
        "Choose data source:",
        options=["realtime_recent", "historical"],
        format_func=lambda x: "🔴 Real-time + Recent Data" if x == "realtime_recent" else "📚 Historical Database",
        key="data_source_radio",
        help="Real-time combines live data with current session history. Historical allows browsing past sessions."
    )
    
    st.session_state.data_source = data_source
    
    # Historical session selection
    if data_source == "historical":
        if st.button("🔄 Refresh Sessions", help="Load latest session list"):
            st.session_state.available_sessions = st.session_state.supabase_manager.get_available_sessions()
        
        if not st.session_state.available_sessions:
            st.session_state.available_sessions = st.session_state.supabase_manager.get_available_sessions()
        
        if st.session_state.available_sessions:
            # Create session selection dropdown
            session_options = []
            for session in st.session_state.available_sessions:
                session_id_short = session['session_id'][:8] + "..."
                start_time = session['start_time']
                record_count = session['record_count']
                session_options.append(f"{session_id_short} | {start_time} | {record_count:,} records")
            
            selected_index = st.selectbox(
                "Select historical session:",
                options=range(len(session_options)),
                format_func=lambda x: session_options[x],
                key="historical_session_select"
            )
            
            st.session_state.selected_historical_session = st.session_state.available_sessions[selected_index]
            
            # Display selected session info
            if st.session_state.selected_historical_session:
                render_session_info(st.session_state.selected_historical_session)
        else:
            st.info("No historical sessions found. Start the bridge to create sessions.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_connection_status(subscriber, stats):
    """Renders connection status with enhanced styling."""
    if subscriber and subscriber.is_connected:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Connected to Real-time</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Disconnected</div>',
            unsafe_allow_html=True,
        )

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


def create_optimized_chart(df: pd.DataFrame, chart_func, title: str):
    """Creates an optimized chart with consistent styling."""
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
            text="No speed data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig = px.line(
        df, x="timestamp", y="speed_ms", title="🚗 Vehicle Speed Over Time",
        labels={"speed_ms": "Speed (m/s)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4"]
    )
    return fig


def create_power_chart(df: pd.DataFrame):
    """Generates a power system chart."""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
        vertical_spacing=0.15
    )

    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["voltage_v"], name="Voltage (V)",
                   line=dict(color="#2ca02c", width=2)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["current_a"], name="Current (A)",
                   line=dict(color="#d62728", width=2)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["power_w"], name="Power (W)",
                   line=dict(color="#ff7f0e", width=2)), row=2, col=1
    )

    fig.update_layout(height=500, title_text="⚡ Electrical System Performance")
    return fig


def create_imu_chart(df: pd.DataFrame):
    """Generates IMU sensor chart."""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
        vertical_spacing=0.25
    )

    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                       line=dict(color=colors_gyro[i], width=2)), row=1, col=1
        )

    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                       line=dict(color=colors_accel[i], width=2)), row=2, col=1
        )

    fig.update_layout(height=600, title_text="🎮 IMU Sensor Data Analysis")
    return fig


def create_imu_chart_2(df: pd.DataFrame):
    """Generates detailed IMU chart with individual subplots."""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("🌀 Gyro X", "🌀 Gyro Y", "🌀 Gyro Z", "📊 Accel X", "📊 Accel Y", "📊 Accel Z"),
        vertical_spacing=0.3, horizontal_spacing=0.1
    )

    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]

    for i, (axis, color) in enumerate(zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                       line=dict(color=color, width=2), showlegend=False), row=1, col=i + 1
        )

    for i, (axis, color) in enumerate(zip(["accel_x", "accel_y", "accel_z"], accel_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                       line=dict(color=color, width=2), showlegend=False), row=2, col=i + 1
        )

    fig.update_layout(height=600, title_text="🎮 Detailed IMU Sensor Analysis")
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Generates efficiency analysis chart."""
    if df.empty or not all(col in df.columns for col in ["speed_ms", "power_w"]):
        return go.Figure().add_annotation(
            text="No efficiency data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig = px.scatter(
        df, x="speed_ms", y="power_w", color="voltage_v" if "voltage_v" in df.columns else None,
        title="⚡ Efficiency Analysis: Speed vs Power Consumption",
        labels={"speed_ms": "Speed (m/s)", "power_w": "Power (W)"},
        color_continuous_scale="viridis"
    )
    return fig


def create_gps_map(df: pd.DataFrame):
    """Generates GPS tracking map."""
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

    fig = px.scatter_map(
        df_valid, lat="latitude", lon="longitude",
        color="speed_ms" if "speed_ms" in df_valid.columns else None,
        size="power_w" if "power_w" in df_valid.columns else None,
        hover_data=["speed_ms", "power_w", "voltage_v"] if all(col in df_valid.columns for col in ["speed_ms", "power_w", "voltage_v"]) else None,
        map_style="open-street-map", title="🛰️ Vehicle Track and Performance",
        height=400, zoom=15, center=center_point, color_continuous_scale="plasma"
    )
    return fig


def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Gets available numeric columns for charting."""
    if df.empty:
        return []
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Creates a dynamic chart based on configuration."""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    if not y_col or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
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
                fig = px.imshow(corr_matrix, title="🔥 Correlation Heatmap", color_continuous_scale="RdBu_r", aspect="auto")
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])

        fig.update_layout(height=400)
        return fig

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )


def render_dynamic_charts_section(df: pd.DataFrame):
    """Renders the dynamic charts section with enhanced UI."""
    st.markdown("### 🎨 Custom Chart Builder")
    
    # Enhanced instructions
    st.markdown("""
    <div class="data-source-selector">
        <h4>🎯 Create Custom Visualizations</h4>
        <p>Build your own charts by selecting variables and chart types. Perfect for exploring correlations and patterns in your telemetry data.</p>
    </div>
    """, unsafe_allow_html=True)

    available_columns = get_available_columns(df)
    
    if not available_columns:
        st.warning("⏳ No numeric data available for creating charts. Please ensure data is loaded.")
        return

    # Chart management controls
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart"):
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

    # Render dynamic charts
    if st.session_state.dynamic_charts:
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Chart configuration controls
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])

                with col1:
                    new_title = st.text_input(
                        "Chart Title", value=chart_config.get("title", "New Chart"),
                        key=f"title_{chart_config['id']}"
                    )
                    st.session_state.dynamic_charts[i]["title"] = new_title

                with col2:
                    new_type = st.selectbox(
                        "Chart Type", options=["line", "scatter", "bar", "histogram", "heatmap"],
                        index=["line", "scatter", "bar", "histogram", "heatmap"].index(chart_config.get("chart_type", "line")),
                        key=f"type_{chart_config['id']}"
                    )
                    st.session_state.dynamic_charts[i]["chart_type"] = new_type

                with col3:
                    if new_type not in ["histogram", "heatmap"]:
                        x_options = (["timestamp"] + available_columns if "timestamp" in df.columns else available_columns)
                        current_x = chart_config.get("x_axis", x_options[0])
                        if current_x not in x_options and x_options:
                            current_x = x_options[0]
                        
                        new_x = st.selectbox(
                            "X-Axis", options=x_options,
                            index=x_options.index(current_x) if current_x in x_options else 0,
                            key=f"x_{chart_config['id']}"
                        )
                        st.session_state.dynamic_charts[i]["x_axis"] = new_x

                with col4:
                    if new_type != "heatmap":
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
                    if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"):
                        st.session_state.dynamic_charts.pop(i)
                        st.rerun()

                # Render the chart
                if chart_config.get("chart_type") == "heatmap" or chart_config.get("y_axis"):
                    fig = create_dynamic_chart(df, chart_config)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config['id']}")
                else:
                    st.warning("Please select a Y-axis variable for this chart.")
                
                st.markdown('</div>', unsafe_allow_html=True)


def load_data_based_on_source():
    """Loads data based on the selected data source."""
    triangulator = st.session_state.data_triangulator
    supabase_manager = st.session_state.supabase_manager
    
    if st.session_state.data_source == "realtime_recent":
        # Real-time + Recent data mode
        realtime_data = []
        supabase_data = []
        history_data = st.session_state.telemetry_data.to_dict('records') if not st.session_state.telemetry_data.empty else []
        
        # Get real-time messages
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            new_messages = st.session_state.subscriber.get_messages()
            if new_messages:
                realtime_data = new_messages
                # Update current session ID from real-time data
                if realtime_data:
                    st.session_state.current_session_id = realtime_data[0].get('session_id')
        
        # Get current session data from Supabase
        if st.session_state.current_session_id:
            supabase_data = supabase_manager.get_current_session_data(st.session_state.current_session_id)
        
        # Triangulate data sources
        combined_data = triangulator.triangulate_data_sources(realtime_data, supabase_data, history_data)
        
        # Update session info
        if st.session_state.current_session_id and combined_data:
            st.session_state.session_info = {
                'session_id': st.session_state.current_session_id,
                'start_time': min(item.get('timestamp', '') for item in combined_data),
                'end_time': max(item.get('timestamp', '') for item in combined_data),
                'record_count': len(combined_data),
                'duration': timedelta(0)  # Will be calculated from start/end times
            }
        
        return combined_data
    
    elif st.session_state.data_source == "historical":
        # Historical data mode
        if st.session_state.selected_historical_session:
            session_id = st.session_state.selected_historical_session['session_id']
            historical_data = supabase_manager.get_historical_session_data(session_id)
            
            # Update session info
            st.session_state.session_info = st.session_state.selected_historical_session
            
            return historical_data
        else:
            return []
    
    return []


def main():
    """Main dashboard application."""
    # Render header
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    initialize_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Connection & Data Control")
        
        # Data source selection
        render_data_source_selector()
        
        st.divider()
        
        # Real-time connection controls (only for real-time mode)
        if st.session_state.data_source == "realtime_recent":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect", use_container_width=True):
                    if st.session_state.subscriber:
                        st.session_state.subscriber.disconnect()
                        time.sleep(1)

                    with st.spinner("Connecting..."):
                        st.session_state.subscriber = TelemetrySubscriber()
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

            # Connection status
            stats = (
                st.session_state.subscriber.get_stats() if st.session_state.subscriber
                else {"messages_received": 0, "connection_attempts": 0, "errors": 0, "last_message_time": None, "last_error": None}
            )
            render_connection_status(st.session_state.subscriber, stats)

            if stats["last_error"]:
                st.error(f"⚠️ {stats['last_error'][:50]}...")

            st.divider()

            # Auto-refresh settings
            st.subheader("⚙️ Settings")
            st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
            
            if st.session_state.auto_refresh:
                refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)
        else:
            st.info("📚 Historical mode selected - no real-time connection needed")

        st.info(f"📡 Channel: {CHANNEL_NAME}")
        st.info(f"🏛️ Database: {SUPABASE_TABLE_NAME}")

    # Load data based on source
    try:
        data_list = load_data_based_on_source()
        
        if data_list:
            df = pd.DataFrame(data_list)
            
            # Ensure timestamp is properly formatted
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Limit datapoints to prevent memory issues
            if len(df) > MAX_DATAPOINTS:
                df = df.tail(MAX_DATAPOINTS)
            
            st.session_state.telemetry_data = df
            st.session_state.last_update = datetime.now()
        else:
            df = pd.DataFrame()
            st.session_state.telemetry_data = df
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()

    # Display session info
    if st.session_state.session_info:
        render_session_info(st.session_state.session_info)

    # Main content area
    if df.empty:
        st.warning("⏳ No telemetry data available.")
        
        if st.session_state.data_source == "realtime_recent":
            col1, col2 = st.columns(2)
            with col1:
                st.info(
                    "**Getting Started:**\n"
                    "1. Ensure m1.py bridge is running\n"
                    "2. Click 'Connect' to start receiving data\n"
                    "3. Wait for telemetry data to appear"
                )
            with col2:
                with st.expander("🔍 Debug Information"):
                    st.json({
                        "Data Source": st.session_state.data_source,
                        "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                        "Session ID": st.session_state.current_session_id,
                        "Available Sessions": len(st.session_state.available_sessions),
                    })
        else:
            st.info("Select a historical session from the sidebar to view data.")
        
        return

    # Display data status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📊 **{len(df):,}** data points loaded")
    with col2:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col3:
        data_source_label = "🔴 Real-time" if st.session_state.data_source == "realtime_recent" else "📚 Historical"
        st.info(f"📡 Source: **{data_source_label}**")

    # Calculate KPIs
    kpis = calculate_kpis(df)

    # Render dashboard tabs
    st.subheader("📈 Dashboard")
    
    tab_names = [
        "📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", "🎮 IMU Detail", 
        "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data"
    ]
    tabs = st.tabs(tab_names)

    # Overview tab
    with tabs[0]:
        st.markdown("### 📊 Performance Overview")
        st.markdown("Real-time key performance indicators for your Shell Eco-marathon vehicle")
        
        # KPI display
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

    # Speed tab
    with tabs[1]:
        render_kpi_header(kpis)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_optimized_chart(df, create_speed_chart, "Speed Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Power tab
    with tabs[2]:
        render_kpi_header(kpis)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_optimized_chart(df, create_power_chart, "Power Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # IMU tab
    with tabs[3]:
        render_kpi_header(kpis)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_optimized_chart(df, create_imu_chart, "IMU Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # IMU Detail tab
    with tabs[4]:
        render_kpi_header(kpis)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_optimized_chart(df, create_imu_chart_2, "IMU Detail Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Efficiency tab
    with tabs[5]:
        render_kpi_header(kpis)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_optimized_chart(df, create_efficiency_chart, "Efficiency Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # GPS tab
    with tabs[6]:
        render_kpi_header(kpis)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Custom tab
    with tabs[7]:
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)

    # Data tab
    with tabs[8]:
        render_kpi_header(kpis)
        st.subheader("📃 Raw Telemetry Data")
        st.warning("ℹ️ Only the **last 100 datapoints** are displayed below. Download the CSV for the complete dataset.")
        
        # Display data table
        display_df = df.tail(100)
        st.dataframe(display_df, use_container_width=True, height=400)

        # Download functionality
        csv = df.to_csv(index=False)
        filename = f"telemetry_{st.session_state.data_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    # Auto-refresh implementation
    if (st.session_state.data_source == "realtime_recent" and 
        st.session_state.auto_refresh and 
        st.session_state.subscriber and 
        st.session_state.subscriber.is_connected):
        
        time.sleep(refresh_interval)
        st.rerun()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>
        <p><strong>Shell Eco-marathon Telemetry Dashboard v2.0</strong> | Enhanced Real-time Data Visualization & Analysis</p>
        <p>🚗 Optimized for performance monitoring and energy efficiency analysis | 
        🔗 <a href="https://github.com/ChosF/EcoTele" target="_blank">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
