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
from typing import Dict, Any, List, Optional
import threading
import queue
import asyncio
import uuid
import hashlib

# Disable tracemalloc warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Handle imports with error checking
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

# Configure Streamlit page
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ChosF/EcoTele",
        "Report a bug": "https://github.com/ChosF/EcoTele/issues",
        "About": "Shell Eco-marathon Real-time Telemetry Dashboard with Supabase Integration",
    },
)

# Setup logging
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger("TelemetrySubscriber")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

setup_terminal_logging()

# Configuration
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
ABLY_CHANNEL_NAME = "telemetry-dashboard-channel"
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"

# Apply enhanced CSS styling
st.markdown("""
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
        --shadow-light: 0 2px 8px rgba(0,0,0,0.1);
        --shadow-medium: 0 4px 16px rgba(0,0,0,0.15);
        --hover-transform: translateY(-2px);
    }

    /* Dark theme overrides */
    [data-theme="dark"] {
        --text-primary: #fafafa;
        --text-secondary: #a0a0a0;
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --border-color: #4a4a4a;
        --shadow-light: 0 2px 8px rgba(255,255,255,0.1);
        --shadow-medium: 0 4px 16px rgba(255,255,255,0.15);
    }

    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: var(--shadow-light);
        background: linear-gradient(135deg, var(--primary-color), #4a90e2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .data-source-card {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .data-source-card:hover {
        border-color: var(--primary-color);
        transform: var(--hover-transform);
        box-shadow: var(--shadow-medium);
    }

    .data-source-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--success-color));
        border-radius: 12px 12px 0 0;
    }

    .data-source-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .data-source-description {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .session-info-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid var(--primary-color);
        box-shadow: var(--shadow-medium);
    }

    .session-info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .session-info-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.95rem;
        color: var(--text-primary);
    }

    .session-info-icon {
        font-size: 1.2rem;
        color: var(--primary-color);
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
        position: relative;
        overflow: hidden;
    }

    .status-indicator::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }

    .status-indicator:hover::before {
        left: 100%;
    }

    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }

    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }

    .status-loading {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 2px solid #ffc107;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        border-color: var(--primary-color);
        transform: var(--hover-transform);
        box-shadow: var(--shadow-medium);
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: var(--primary-color);
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .chart-container {
        background: var(--bg-primary);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid var(--border-color);
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }

    .chart-container:hover {
        box-shadow: var(--shadow-medium);
    }

    .custom-chart-section {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 6px solid var(--primary-color);
        box-shadow: var(--shadow-medium);
    }

    .instructions-title {
        color: var(--primary-color);
        font-size: 1.5rem;
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
        margin-bottom: 1.5rem;
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
        padding: 1.5rem;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        text-align: center;
    }

    .chart-type-card:hover {
        border-color: var(--primary-color);
        transform: var(--hover-transform);
        box-shadow: var(--shadow-medium);
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

    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 50;
        background: var(--bg-primary);
        border-bottom: 2px solid var(--border-color);
        border-radius: 12px 12px 0 0;
        padding: 0.75rem;
        box-shadow: var(--shadow-light);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-secondary);
        border-color: var(--primary-color);
        transform: translateY(-1px);
    }

    .stButton > button {
        border-radius: 10px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 0.75rem 1.5rem;
    }

    .stButton > button:hover {
        background: transparent;
        color: var(--primary-color);
        transform: var(--hover-transform);
        box-shadow: var(--shadow-medium);
    }

    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: var(--primary-color);
        box-shadow: var(--shadow-light);
    }

    /* Enhanced scrollbar styling */
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
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #4a90e2;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }

        .metrics-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }

        .chart-type-grid {
            grid-template-columns: 1fr;
        }

        .session-info-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Loading animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)


class DataTriangulation:
    """Handles data deduplication and triangulation across multiple sources"""
    
    @staticmethod
    def create_data_fingerprint(row: dict) -> str:
        """Create a unique fingerprint for a data row based on timestamp and key metrics"""
        key_fields = ['timestamp', 'speed_ms', 'voltage_v', 'current_a', 'latitude', 'longitude']
        fingerprint_data = {}
        
        for field in key_fields:
            if field in row:
                value = row[field]
                if isinstance(value, float):
                    # Round floats to avoid precision issues
                    fingerprint_data[field] = round(value, 6)
                else:
                    fingerprint_data[field] = value
        
        # Create hash of the fingerprint data
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    @staticmethod
    def merge_data_sources(realtime_data: pd.DataFrame, supabase_data: pd.DataFrame, 
                          streamlit_history: pd.DataFrame) -> pd.DataFrame:
        """Merge multiple data sources with deduplication"""
        
        # Combine all data sources
        all_dataframes = []
        
        # Add source tags
        if not realtime_data.empty:
            realtime_data = realtime_data.copy()
            realtime_data['data_source_type'] = 'realtime'
            all_dataframes.append(realtime_data)
        
        if not supabase_data.empty:
            supabase_data = supabase_data.copy()
            supabase_data['data_source_type'] = 'supabase'
            all_dataframes.append(supabase_data)
        
        if not streamlit_history.empty:
            streamlit_history = streamlit_history.copy()
            streamlit_history['data_source_type'] = 'streamlit_history'
            all_dataframes.append(streamlit_history)
        
        if not all_dataframes:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Create fingerprints for deduplication
        combined_df['fingerprint'] = combined_df.apply(
            lambda row: DataTriangulation.create_data_fingerprint(row.to_dict()), 
            axis=1
        )
        
        # Sort by timestamp and remove duplicates, keeping the most recent source
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values(['timestamp', 'data_source_type'])
        
        # Prioritize data sources: realtime > supabase > streamlit_history
        source_priority = {'realtime': 3, 'supabase': 2, 'streamlit_history': 1}
        combined_df['source_priority'] = combined_df['data_source_type'].map(source_priority)
        
        # Remove duplicates based on fingerprint, keeping highest priority
        combined_df = combined_df.sort_values(['fingerprint', 'source_priority'], ascending=[True, False])
        deduplicated_df = combined_df.drop_duplicates(subset='fingerprint', keep='first')
        
        # Clean up temporary columns
        deduplicated_df = deduplicated_df.drop(['fingerprint', 'source_priority'], axis=1)
        
        # Sort by timestamp for final output
        deduplicated_df = deduplicated_df.sort_values('timestamp').reset_index(drop=True)
        
        return deduplicated_df


class SupabaseDataManager:
    """Manages Supabase database operations"""
    
    def __init__(self):
        self.supabase_client = None
        self.logger = logging.getLogger("SupabaseDataManager")
        self._connect()
    
    def _connect(self):
        """Connect to Supabase"""
        try:
            self.supabase_client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
            self.logger.info("✅ Connected to Supabase")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Supabase: {e}")
            self.supabase_client = None
    
    def get_available_sessions(self) -> List[Dict[str, Any]]:
        """Get list of available sessions from database"""
        if not self.supabase_client:
            return []
        
        try:
            # Get unique sessions with their metadata
            response = self.supabase_client.table(SUPABASE_TABLE_NAME).select(
                "session_id, timestamp"
            ).execute()
            
            if not response.data:
                return []
            
            # Group by session_id and calculate metadata
            sessions_df = pd.DataFrame(response.data)
            sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
            
            session_info = []
            for session_id in sessions_df['session_id'].unique():
                session_data = sessions_df[sessions_df['session_id'] == session_id]
                
                start_time = session_data['timestamp'].min()
                end_time = session_data['timestamp'].max()
                duration = end_time - start_time
                record_count = len(session_data)
                
                session_info.append({
                    'session_id': session_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'record_count': record_count
                })
            
            # Sort by start time (most recent first)
            session_info.sort(key=lambda x: x['start_time'], reverse=True)
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching sessions: {e}")
            return []
    
    def get_session_data(self, session_id: str) -> pd.DataFrame:
        """Get all data for a specific session"""
        if not self.supabase_client:
            return pd.DataFrame()
        
        try:
            response = self.supabase_client.table(SUPABASE_TABLE_NAME).select("*").eq(
                "session_id", session_id
            ).order("timestamp", desc=False).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching session data: {e}")
            return pd.DataFrame()
    
    def get_current_session_data(self) -> pd.DataFrame:
        """Get data from the most recent session"""
        sessions = self.get_available_sessions()
        if not sessions:
            return pd.DataFrame()
        
        # Get the most recent session
        current_session = sessions[0]
        return self.get_session_data(current_session['session_id'])


class AblyHistoryManager:
    """Manages Ably history retrieval"""
    
    def __init__(self):
        self.ably_client = None
        self.channel = None
        self.logger = logging.getLogger("AblyHistoryManager")
    
    async def connect(self):
        """Connect to Ably"""
        try:
            self.ably_client = AblyRealtime(ABLY_API_KEY)
            self.channel = self.ably_client.channels.get(ABLY_CHANNEL_NAME)
            self.logger.info("✅ Connected to Ably for history retrieval")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Ably: {e}")
            return False
    
    async def get_history(self, limit: int = 1000, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get message history from Ably"""
        if not self.channel:
            return pd.DataFrame()
        
        try:
            # Prepare history parameters
            params = {
                'limit': min(limit, 1000),  # Ably max limit is 1000
                'direction': 'backwards'
            }
            
            if start_time:
                params['start'] = int(start_time.timestamp() * 1000)  # Convert to milliseconds
            
            # Get history
            history = await self.channel.history(params)
            
            messages = []
            for message in history.items:
                if message.name == 'telemetry_update' and message.data:
                    messages.append(message.data)
            
            if not messages:
                return pd.DataFrame()
            
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching Ably history: {e}")
            return pd.DataFrame()
    
    async def close(self):
        """Close Ably connection"""
        if self.ably_client:
            await self.ably_client.close()


class TelemetrySubscriber:
    """Manages real-time Ably connection and message subscription"""
    
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
            
            self.connection_thread = threading.Thread(target=self._connection_worker, daemon=True)
            self.connection_thread.start()
            
            time.sleep(3)  # Allow connection to establish
            
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
            
            self.logger.info(f"📡 Getting channel: {ABLY_CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(ABLY_CHANNEL_NAME)
            
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
                # Prevent queue overflow
                if self.message_queue.qsize() > 1000:
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
        "supabase_manager": SupabaseDataManager(),
        "ably_history_manager": AblyHistoryManager(),
        "telemetry_data": pd.DataFrame(),
        "streamlit_history": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "data_source_mode": "realtime_recent",  # or "historical"
        "selected_session": None,
        "dynamic_charts": [],
        "current_session_info": None,
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
        # Ensure numeric columns
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
                    gyro_data["gyro_x"] ** 2 + gyro_data["gyro_y"] ** 2 + gyro_data["gyro_z"] ** 2
                )
                kpis["avg_gyro_magnitude"] = max(0, gyro_magnitude.mean())
        
        return kpis
        
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis


def render_kpi_header(kpis: Dict[str, float]):
    """Render KPI header with enhanced styling"""
    st.markdown("""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-icon">📏</div>
            <div class="metric-value">{:.2f} km</div>
            <div class="metric-label">Total Distance</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🚀</div>
            <div class="metric-value">{:.1f} m/s</div>
            <div class="metric-label">Max Speed</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🔋</div>
            <div class="metric-value">{:.2f} MJ</div>
            <div class="metric-label">Total Energy</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{:.1f} W</div>
            <div class="metric-label">Avg Power</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">♻️</div>
            <div class="metric-value">{:.2f} km/MJ</div>
            <div class="metric-label">Efficiency</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{:.2f} m/s²</div>
            <div class="metric-label">Max Acceleration</div>
        </div>
    </div>
    """.format(
        kpis['total_distance_km'],
        kpis['max_speed_ms'],
        kpis['total_energy_mj'],
        kpis['avg_power_w'],
        kpis['efficiency_km_per_mj'],
        kpis['max_acceleration']
    ), unsafe_allow_html=True)


def render_session_info(session_info: Dict[str, Any]):
    """Render session information card"""
    if not session_info:
        return
    
    session_id_short = session_info['session_id'][:8] + "..." if len(session_info['session_id']) > 8 else session_info['session_id']
    start_time_str = session_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    duration_str = str(session_info['duration'])
    
    st.markdown(f"""
    <div class="session-info-card">
        <div class="instructions-title">
            📊 Current Session Information
        </div>
        <div class="session-info-grid">
            <div class="session-info-item">
                <span class="session-info-icon">📋</span>
                <span><strong>Session:</strong> {session_id_short}</span>
            </div>
            <div class="session-info-item">
                <span class="session-info-icon">📅</span>
                <span><strong>Start:</strong> {start_time_str}</span>
            </div>
            <div class="session-info-item">
                <span class="session-info-icon">⏱️</span>
                <span><strong>Duration:</strong> {duration_str}</span>
            </div>
            <div class="session-info-item">
                <span class="session-info-icon">📊</span>
                <span><strong>Records:</strong> {session_info['record_count']:,}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_data_source_selection():
    """Render data source selection interface"""
    st.markdown("""
    <div class="custom-chart-section">
        <div class="instructions-title">
            🔧 Data Source Configuration
        </div>
        <div class="instructions-content">
            Choose how you want to view your telemetry data. The system combines multiple sources 
            to provide comprehensive coverage with intelligent deduplication.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="data-source-card">
            <div class="data-source-title">🔴 Real-time + Recent Data</div>
            <div class="data-source-description">
                Combines live telemetry, recent Ably history, and current session data from 
                Supabase. Best for monitoring ongoing sessions with full historical context.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔴 Use Real-time + Recent", key="realtime_mode", use_container_width=True):
            st.session_state.data_source_mode = "realtime_recent"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="data-source-card">
            <div class="data-source-title">🗂️ Historical Session Data</div>
            <div class="data-source-description">
                View complete historical data from any previous session. Perfect for analysis, 
                comparison, and reviewing past performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗂️ Use Historical Data", key="historical_mode", use_container_width=True):
            st.session_state.data_source_mode = "historical"
            st.rerun()
    
    # Show current mode
    if st.session_state.data_source_mode == "realtime_recent":
        st.success("🔴 **Current Mode:** Real-time + Recent Data")
    else:
        st.info("🗂️ **Current Mode:** Historical Session Data")


def render_connection_status(subscriber, stats):
    """Render connection status in sidebar"""
    if subscriber and subscriber.is_connected:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Real-time Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Real-time Disconnected</div>',
            unsafe_allow_html=True,
        )
    
    # Database status
    if st.session_state.supabase_manager.supabase_client:
        st.sidebar.markdown(
            '<div class="status-indicator status-connected">✅ Database Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-indicator status-disconnected">❌ Database Disconnected</div>',
            unsafe_allow_html=True,
        )
    
    # Stats
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


def create_speed_chart(df: pd.DataFrame):
    """Create speed chart"""
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
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        title=dict(font=dict(size=16, color="#1f77b4")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    
    return fig


def create_power_chart(df: pd.DataFrame):
    """Create power chart with voltage, current, and power"""
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
        return go.Figure().add_annotation(
            text="No power data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("⚡ Voltage & Current", "🔋 Power Output"),
        vertical_spacing=0.15,
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
    
    fig.update_layout(
        height=500, title_text="⚡ Electrical System Performance",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig


def create_imu_chart(df: pd.DataFrame):
    """Create IMU chart"""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
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
                      line=dict(color=colors_gyro[i], width=2)), row=1, col=1
        )
    
    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                      line=dict(color=colors_accel[i], width=2)), row=2, col=1
        )
    
    fig.update_layout(
        height=600, title_text="🎮 IMU Sensor Data Analysis",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Create efficiency chart"""
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
        color_continuous_scale="viridis",
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        title=dict(font=dict(size=16, color="#1f77b4")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    
    return fig


def create_gps_map(df: pd.DataFrame):
    """Create GPS map"""
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
        map_style="open-street-map",
        title="🛰️ Vehicle Track and Performance",
        height=400, zoom=15, center=center_point,
        color_continuous_scale="plasma",
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
        
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            title=dict(font=dict(size=16, color="#1f77b4")),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )


def render_custom_charts_section(df: pd.DataFrame):
    """Render custom charts section"""
    st.markdown("""
    <div class="custom-chart-section">
        <div class="instructions-title">
            🎯 Create Custom Charts
        </div>
        <div class="instructions-content">
            <p>Create personalized visualizations by selecting your preferred variables and chart types.</p>
            <p><strong>Note:</strong> Chart visibility may be reduced when auto-refresh is enabled.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart type information
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
        st.warning("⏳ No numeric data available for creating charts. Please ensure data is loaded.")
        return
    
    # Chart management
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
    
    # Render charts
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
                            x_options = (["timestamp"] + available_columns
                                       if "timestamp" in df.columns else available_columns)
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
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"):
                            try:
                                st.session_state.dynamic_charts.pop(i)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")
                    
                    # Render chart
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
    # Header
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar
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
        stats = (st.session_state.subscriber.get_stats() if st.session_state.subscriber
                else {"messages_received": 0, "connection_attempts": 0, "errors": 0,
                      "last_message_time": None, "last_error": None})
        
        render_connection_status(st.session_state.subscriber, stats)
        
        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        new_auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        if new_auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = new_auto_refresh
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)
        
        st.info(f"📡 Channel: {ABLY_CHANNEL_NAME}")
    
    # Data source selection
    render_data_source_selection()
    
    # Handle data loading based on mode
    if st.session_state.data_source_mode == "historical":
        # Historical mode - session selection
        sessions = st.session_state.supabase_manager.get_available_sessions()
        
        if not sessions:
            st.warning("⏳ No historical sessions found in the database.")
            return
        
        # Session selection
        session_options = []
        for session in sessions:
            session_short = session['session_id'][:8] + "..."
            start_time = session['start_time'].strftime('%Y-%m-%d %H:%M:%S')
            session_options.append(f"{session_short} - {start_time} ({session['record_count']} records)")
        
        selected_session_idx = st.selectbox(
            "📋 Select Historical Session",
            range(len(session_options)),
            format_func=lambda x: session_options[x]
        )
        
        selected_session = sessions[selected_session_idx]
        
        # Load session data
        with st.spinner("Loading session data..."):
            df = st.session_state.supabase_manager.get_session_data(selected_session['session_id'])
        
        if df.empty:
            st.error("❌ No data found for selected session.")
            return
        
        # Display session info
        render_session_info(selected_session)
        
    else:
        # Real-time + Recent mode
        new_messages_count = 0
        
        # Get real-time messages
        realtime_data = pd.DataFrame()
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            new_messages = st.session_state.subscriber.get_messages()
            
            if new_messages:
                new_messages_count = len(new_messages)
                realtime_data = pd.DataFrame(new_messages)
                
                if "timestamp" in realtime_data.columns:
                    realtime_data["timestamp"] = pd.to_datetime(realtime_data["timestamp"])
                
                # Update Streamlit history
                if st.session_state.streamlit_history.empty:
                    st.session_state.streamlit_history = realtime_data
                else:
                    st.session_state.streamlit_history = pd.concat(
                        [st.session_state.streamlit_history, realtime_data], ignore_index=True
                    )
                
                # Keep only last 10,000 records in memory
                if len(st.session_state.streamlit_history) > 10000:
                    st.session_state.streamlit_history = st.session_state.streamlit_history.tail(10000)
                
                st.session_state.last_update = datetime.now()
        
        # Get Supabase current session data
        with st.spinner("Loading current session data..."):
            supabase_data = st.session_state.supabase_manager.get_current_session_data()
        
        # Merge all data sources using triangulation
        df = DataTriangulation.merge_data_sources(
            realtime_data, supabase_data, st.session_state.streamlit_history
        )
        
        # Create current session info
        if not df.empty:
            current_session_info = {
                'session_id': df['session_id'].iloc[0] if 'session_id' in df.columns else 'unknown',
                'start_time': df['timestamp'].min(),
                'end_time': df['timestamp'].max(),
                'duration': df['timestamp'].max() - df['timestamp'].min(),
                'record_count': len(df)
            }
            render_session_info(current_session_info)
    
    # Show data status
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Getting Started:**\n1. Ensure m1.py bridge is running\n2. Click 'Connect' to start receiving data")
        
        with col2:
            with st.expander("🔍 Debug Information"):
                st.json({
                    "Real-time Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                    "Database Connected": st.session_state.supabase_manager.supabase_client is not None,
                    "Messages": stats["messages_received"],
                    "Errors": stats["errors"],
                    "Mode": st.session_state.data_source_mode,
                })
        return
    
    # Data status
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        data_sources = df['data_source_type'].unique() if 'data_source_type' in df.columns else ['unknown']
        st.info(f"📊 **{len(df):,}** data points from {len(data_sources)} sources")
    with col2:
        st.info(f"⏰ Data range: **{df['timestamp'].min().strftime('%H:%M:%S')} - {df['timestamp'].max().strftime('%H:%M:%S')}**")
    with col3:
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Main dashboard tabs
    st.subheader("📈 Dashboard")
    
    tab_names = ["📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data"]
    tabs = st.tabs(tab_names)
    
    # Overview tab
    with tabs[0]:
        render_kpi_header(kpis)
        
        # Data source breakdown
        if 'data_source_type' in df.columns:
            st.subheader("📊 Data Source Breakdown")
            source_counts = df['data_source_type'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            for i, (source, count) in enumerate(source_counts.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(f"🔹 {source.title()}", f"{count:,} records")
    
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
        render_custom_charts_section(df)
    
    # Data tab
    with tabs[7]:
        render_kpi_header(kpis)
        
        st.subheader("📃 Raw Telemetry Data")
        
        # Data filtering options
        col1, col2 = st.columns(2)
        with col1:
            show_records = st.slider("Show last N records", 100, min(5000, len(df)), 500)
        with col2:
            if 'data_source_type' in df.columns:
                source_filter = st.multiselect(
                    "Filter by data source",
                    options=df['data_source_type'].unique(),
                    default=df['data_source_type'].unique()
                )
                if source_filter:
                    df_filtered = df[df['data_source_type'].isin(source_filter)]
                else:
                    df_filtered = df
            else:
                df_filtered = df
        
        display_df = df_filtered.tail(show_records)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"telemetry_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col2:
            csv_all = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Data (CSV)",
                data=csv_all,
                file_name=f"telemetry_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    # Auto-refresh
    if (st.session_state.auto_refresh and st.session_state.data_source_mode == "realtime_recent" 
        and st.session_state.subscriber and st.session_state.subscriber.is_connected):
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: var(--text-secondary); padding: 1rem;'>
        <p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | Enhanced with Supabase Integration</p>
        <p>🚗 Real-time monitoring • 💾 Historical analysis • 📊 Advanced data triangulation</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
