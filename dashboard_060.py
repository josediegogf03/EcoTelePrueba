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
import math

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

# Pagination constants
SUPABASE_MAX_ROWS_PER_REQUEST = 1000
MAX_DATAPOINTS_PER_SESSION = 1000000

# Configures the Streamlit page
st.set_page_config(
    page_title="🏎️ Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Shell Eco-marathon Telemetry Dashboard",
    },
)

# CSS styling
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

    .pagination-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #0d47a1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #2196f3;
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
        background: transparent;
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
    """Telemetry manager with multi-source data integration and pagination support."""
    
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
            "data_sources": [],
            "pagination_stats": {
                "total_requests": 0,
                "total_rows_fetched": 0,
                "largest_session_size": 0,
                "sessions_paginated": 0
            }
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
                if self.message_queue.qsize() > 500:
                    while self.message_queue.qsize() > 250:
                        try:
                            self.message_queue.get_nowait()
                        except queue.Empty:
                            break
                
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
    
    def _paginated_fetch(self, session_id: str, data_source: str = 'supabase_current') -> pd.DataFrame:
        """Fetch all data for a session using pagination."""
        try:
            if not self.supabase_client:
                self.logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()
            
            all_data = []
            offset = 0
            total_fetched = 0
            request_count = 0
            max_requests = math.ceil(MAX_DATAPOINTS_PER_SESSION / SUPABASE_MAX_ROWS_PER_REQUEST)
            
            self.logger.info(f"🔄 Starting paginated fetch for session {session_id[:8]}...")
            
            while offset < MAX_DATAPOINTS_PER_SESSION:
                try:
                    range_end = offset + SUPABASE_MAX_ROWS_PER_REQUEST - 1
                    
                    response = (self.supabase_client
                              .table(SUPABASE_TABLE_NAME)
                              .select("*")
                              .eq("session_id", session_id)
                              .order("timestamp", desc=False)
                              .range(offset, range_end)
                              .execute())
                    
                    request_count += 1
                    
                    if not response.data:
                        break
                    
                    batch_size = len(response.data)
                    all_data.extend(response.data)
                    total_fetched += batch_size
                    
                    if batch_size < SUPABASE_MAX_ROWS_PER_REQUEST:
                        break
                    
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in pagination request {request_count}: {e}")
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    continue
            
            with self._lock:
                self.stats["pagination_stats"]["total_requests"] += request_count
                self.stats["pagination_stats"]["total_rows_fetched"] += total_fetched
                self.stats["pagination_stats"]["largest_session_size"] = max(
                    self.stats["pagination_stats"]["largest_session_size"], total_fetched
                )
                if request_count > 1:
                    self.stats["pagination_stats"]["sessions_paginated"] += 1
            
            if all_data:
                df = pd.DataFrame(all_data)
                df['data_source'] = data_source
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error in paginated fetch for session {session_id}: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return pd.DataFrame()
    
    def get_current_session_data(self, session_id: str) -> pd.DataFrame:
        """Get current session data from Supabase with pagination support."""
        return self._paginated_fetch(session_id, 'supabase_current')
    
    def get_historical_sessions(self) -> List[Dict[str, Any]]:
        """Get list of historical sessions with pagination support."""
        try:
            if not self.supabase_client:
                return []
            
            all_records = []
            offset = 0
            
            while True:
                try:
                    range_end = offset + SUPABASE_MAX_ROWS_PER_REQUEST - 1
                    
                    response = (self.supabase_client
                              .table(SUPABASE_TABLE_NAME)
                              .select("session_id, timestamp")
                              .order("timestamp", desc=True)
                              .range(offset, range_end)
                              .execute())
                    
                    if not response.data:
                        break
                    
                    all_records.extend(response.data)
                    
                    if len(response.data) < SUPABASE_MAX_ROWS_PER_REQUEST:
                        break
                    
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    
                except Exception as e:
                    self.logger.error(f"❌ Error fetching session records: {e}")
                    break
            
            if not all_records:
                return []
            
            sessions = {}
            for record in all_records:
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
            
            session_list = []
            for session_info in sessions.values():
                try:
                    start_dt = datetime.fromisoformat(session_info['start_time'].replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(session_info['end_time'].replace('Z', '+00:00'))
                    duration = end_dt - start_dt
                    
                    session_list.append({
                        'session_id': session_info['session_id'],
                        'star