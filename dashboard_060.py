# dashboard_enhanced.py - Enhanced Shell Eco-marathon Telemetry Dashboard with Multi-Source Data
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
import hashlib
from dataclasses import dataclass
from enum import Enum

# Disables tracemalloc warnings
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
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Shell Eco-marathon Real-time Telemetry Dashboard v2.0",
    },
)

# Enhanced CSS styling for modern dark/light mode compatibility
st.markdown("""
<style>
    /* Enhanced theme-aware color variables */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #f39c12;
        --error-color: #d62728;
        --info-color: #17a2b8;
        --text-primary: #262730;
        --text-secondary: #6c757d;
        --text-muted: #8e9297;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --border-color: #dee2e6;
        --hover-color: #e3f2fd;
        --shadow-light: 0 2px 8px rgba(0,0,0,0.1);
        --shadow-medium: 0 4px 16px rgba(0,0,0,0.15);
        --shadow-heavy: 0 8px 32px rgba(0,0,0,0.2);
    }

    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #fafafa;
            --text-secondary: #b0b0b0;
            --text-muted: #808080;
            --bg-primary: #0e1117;
            --bg-secondary: #1a1d29;
            --bg-tertiary: #262730;
            --border-color: #3a3a3a;
            --hover-color: #1e2329;
            --shadow-light: 0 2px 8px rgba(255,255,255,0.05);
            --shadow-medium: 0 4px 16px rgba(255,255,255,0.1);
            --shadow-heavy: 0 8px 32px rgba(255,255,255,0.15);
        }
    }

    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: var(--shadow-light);
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Enhanced status indicators */
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
        box-shadow: var(--shadow-light);
    }

    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid var(--success-color);
        box-shadow: 0 4px 16px rgba(40, 167, 69, 0.25);
    }

    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid var(--error-color);
        box-shadow: 0 4px 16px rgba(220, 53, 69, 0.25);
    }

    .status-historical {
        background: linear-gradient(135deg, #e2e3e5 0%, #d1d3d4 100%);
        color: #383d41;
        border: 2px solid var(--text-muted);
        box-shadow: 0 4px 16px rgba(56, 61, 65, 0.25);
    }

    /* Session information card */
    .session-card {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid var(--border-color);
        box-shadow: var(--shadow-medium);
        transition: all 0.3s ease;
    }

    .session-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-heavy);
        border-color: var(--primary-color);
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
        color: var(--text-secondary);
    }

    .session-icon {
        font-size: 1.2rem;
        color: var(--primary-color);
    }

    /* Data source selector */
    .data-source-selector {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }

    /* Enhanced metric cards */
    .metric-card {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Warning banner for historical data */
    .historical-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid var(--warning-color);
        box-shadow: var(--shadow-light);
        text-align: center;
        font-weight: 600;
    }

    /* Improved chart containers */
    .chart-container {
        background: var(--bg-primary);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-light);
    }

    /* Enhanced button styling */
    .stButton > button {
        border-radius: 8px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-light);
    }

    .stButton > button:hover {
        background: transparent;
        color: var(--primary-color);
        transform: translateY(-1px);
        box-shadow: var(--shadow-medium);
    }

    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 50;
        background: var(--bg-primary);
        border-bottom: 2px solid var(--border-color);
        border-radius: 8px 8px 0 0;
        padding: 0.5rem;
        box-shadow: var(--shadow-light);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--hover-color);
        transform: translateY(-1px);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .session-info {
            grid-template-columns: 1fr;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }

    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Configuration constants
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"
MAX_DATAPOINTS = 1000000000  # 1 billion points limit
ABLY_HISTORY_LIMIT = 1000  # Ably history API limit per request

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data source types
class DataSourceType(Enum):
    REALTIME_PLUS_SESSION = "realtime_plus_session"
    HISTORICAL = "historical"

@dataclass
class SessionInfo:
    """Session information structure"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: timedelta
    record_count: int
    is_current: bool

class DataTriangulator:
    """Handles data triangulation to prevent duplicates and lost data"""
    
    def __init__(self):
        self.seen_messages = set()
        self.message_hashes = {}
        
    def create_message_hash(self, message: Dict[str, Any]) -> str:
        """Create a unique hash for a message based on timestamp and key fields"""
        key_fields = {
            'timestamp': message.get('timestamp', ''),
            'session_id': message.get('session_id', ''),
            'message_id': message.get('message_id', ''),
            'speed_ms': message.get('speed_ms', 0),
            'voltage_v': message.get('voltage_v', 0),
            'current_a': message.get('current_a', 0),
        }
        
        # Create hash from key fields
        hash_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def is_duplicate(self, message: Dict[str, Any]) -> bool:
        """Check if a message is a duplicate"""
        msg_hash = self.create_message_hash(message)
        if msg_hash in self.seen_messages:
            return True
        
        self.seen_messages.add(msg_hash)
        return False
    
    def merge_data_sources(self, sources: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple data sources, removing duplicates"""
        if not sources:
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(sources, ignore_index=True)
        
        if combined_df.empty:
            return combined_df
        
        # Remove duplicates based on timestamp and key fields
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp')
            combined_df = combined_df.drop_duplicates(
                subset=['timestamp', 'session_id', 'message_id'], 
                keep='first'
            )
        
        return combined_df.reset_index(drop=True)

class EnhancedTelemetrySubscriber:
    """Enhanced telemetry subscriber with multiple data sources"""
    
    def __init__(self):
        self.ably_client = None
        self.supabase_client = None
        self.channel = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.connection_thread = None
        self.current_session_id = None
        self.data_triangulator = DataTriangulator()
        
        # Statistics
        self.stats = {
            "messages_received": 0,
            "ably_history_retrieved": 0,
            "supabase_records_retrieved": 0,
            "last_message_time": None,
            "connection_attempts": 0,
            "errors": 0,
            "last_error": None,
        }
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        
        # Initialize logger
        self.logger = logging.getLogger("EnhancedTelemetrySubscriber")
        
        # Initialize Supabase client
        self._init_supabase()
    
    def _init_supabase(self):
        """Initialize Supabase client"""
        try:
            self.supabase_client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
            self.logger.info("✅ Supabase client initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Supabase client: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
    
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
        """Worker thread for Ably connection"""
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
            
            # Connection event handlers
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
            
            # Wait for connection
            self.logger.info("⏳ Waiting for connection...")
            await self.ably_client.connection.once_async("connected")
            
            # Get channel and subscribe
            self.logger.info(f"📡 Getting channel: {CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(CHANNEL_NAME)
            
            self.logger.info("📨 Subscribing to messages...")
            await self.channel.subscribe("telemetry_update", self._on_message_received)
            
            self.logger.info("✅ Successfully subscribed to messages!")
            
            # Keep connection alive
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
            
            # Parse JSON if needed
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
            
            # Extract current session ID
            if 'session_id' in data:
                self.current_session_id = data['session_id']
            
            # Check for duplicates
            if not self.data_triangulator.is_duplicate(data):
                with self._lock:
                    if self.message_queue.qsize() > 1000:
                        # Remove old messages to prevent memory issues
                        while self.message_queue.qsize() > 500:
                            try:
                                self.message_queue.get_nowait()
                            except queue.Empty:
                                break
                    
                    self.message_queue.put(data)
                    self.stats["messages_received"] += 1
                    self.stats["last_message_time"] = datetime.now()
                    
                    self.logger.debug(f"✅ Message queued. Total: {self.stats['messages_received']}")
            
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Message error: {e}"
    
    async def get_ably_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get message history from Ably"""
        try:
            if not self.channel:
                return []
            
            # Get history with pagination
            history_messages = []
            
            # Get first page
            history = await self.channel.history({"limit": min(limit, 1000)})
            
            if history.items:
                for message in history.items:
                    data = message.data
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                    
                    if isinstance(data, dict):
                        history_messages.append(data)
                
                # Get additional pages if needed
                while history.has_next() and len(history_messages) < limit:
                    try:
                        history = await history.next()
                        if history.items:
                            for message in history.items:
                                data = message.data
                                if isinstance(data, str):
                                    try:
                                        data = json.loads(data)
                                    except json.JSONDecodeError:
                                        continue
                                
                                if isinstance(data, dict):
                                    history_messages.append(data)
                                    
                                    if len(history_messages) >= limit:
                                        break
                        else:
                            break
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error getting next history page: {e}")
                        break
            
            with self._lock:
                self.stats["ably_history_retrieved"] += len(history_messages)
            
            self.logger.info(f"📚 Retrieved {len(history_messages)} messages from Ably history")
            return history_messages
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving Ably history: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Ably history error: {e}"
            return []
    
    def get_supabase_sessions(self) -> List[SessionInfo]:
        """Get all available sessions from Supabase"""
        try:
            if not self.supabase_client:
                return []
            
            # Get session summary
            response = self.supabase_client.table(SUPABASE_TABLE_NAME).select(
                "session_id, timestamp"
            ).order("timestamp", desc=True).execute()
            
            if not response.data:
                return []
            
            # Group by session_id
            sessions_data = {}
            for record in response.data:
                session_id = record['session_id']
                timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                
                if session_id not in sessions_data:
                    sessions_data[session_id] = {
                        'timestamps': [],
                        'count': 0
                    }
                
                sessions_data[session_id]['timestamps'].append(timestamp)
                sessions_data[session_id]['count'] += 1
            
            # Create SessionInfo objects
            sessions = []
            for session_id, data in sessions_data.items():
                timestamps = sorted(data['timestamps'])
                start_time = timestamps[0]
                end_time = timestamps[-1] if len(timestamps) > 1 else None
                duration = (end_time - start_time) if end_time else timedelta(0)
                
                session_info = SessionInfo(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    record_count=data['count'],
                    is_current=(session_id == self.current_session_id)
                )
                sessions.append(session_info)
            
            # Sort by start time (newest first)
            sessions.sort(key=lambda x: x.start_time, reverse=True)
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting Supabase sessions: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Supabase sessions error: {e}"
            return []
    
    def get_supabase_data(self, session_id: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get data from Supabase for a specific session or current session"""
        try:
            if not self.supabase_client:
                return []
            
            query = self.supabase_client.table(SUPABASE_TABLE_NAME).select("*")
            
            # Filter by session if specified
            if session_id:
                query = query.eq("session_id", session_id)
            elif self.current_session_id:
                query = query.eq("session_id", self.current_session_id)
            
            # Apply limit if specified
            if limit:
                query = query.limit(limit)
            
            # Order by timestamp
            query = query.order("timestamp", desc=False)
            
            response = query.execute()
            
            if response.data:
                with self._lock:
                    self.stats["supabase_records_retrieved"] += len(response.data)
                
                self.logger.info(f"💾 Retrieved {len(response.data)} records from Supabase")
                return response.data
            
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving Supabase data: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Supabase data error: {e}"
            return []
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all queued real-time messages"""
        messages = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
        
        if messages:
            self.logger.debug(f"📤 Returning {len(messages)} real-time messages")
        
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
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "data_triangulator": DataTriangulator(),
        "current_data_source": DataSourceType.REALTIME_PLUS_SESSION,
        "selected_session": None,
        "available_sessions": [],
        "is_historical_mode": False,
        "last_data_refresh": datetime.now(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators from the telemetry DataFrame"""
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
        
        # Calculate KPIs
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
        
        # Calculate efficiency
        if kpis["total_energy_mj"] > 0:
            kpis["efficiency_km_per_mj"] = kpis["total_distance_km"] / kpis["total_energy_mj"]
        
        if "total_acceleration" in df.columns:
            accel_data = df["total_acceleration"].dropna()
            if not accel_data.empty:
                kpis["max_acceleration"] = max(0, accel_data.max())
        
        # Calculate gyro magnitude
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

def render_session_card(session: SessionInfo):
    """Render a session information card"""
    status_icon = "🟢" if session.is_current else "🔵"
    
    st.markdown(f"""
    <div class="session-card">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">{status_icon}</span>
            <div>
                <h3 style="margin: 0; color: var(--primary-color);">
                    {'Current Session' if session.is_current else 'Historical Session'}
                </h3>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">
                    {session.session_id}
                </p>
            </div>
        </div>
        
        <div class="session-info">
            <div class="session-detail">
                <span class="session-icon">📋</span>
                <span><strong>Session:</strong> {session.session_id[:8]}...</span>
            </div>
            <div class="session-detail">
                <span class="session-icon">📅</span>
                <span><strong>Start:</strong> {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <div class="session-detail">
                <span class="session-icon">⏱️</span>
                <span><strong>Duration:</strong> {session.duration}</span>
            </div>
            <div class="session-detail">
                <span class="session-icon">📊</span>
                <span><strong>Records:</strong> {session.record_count:,}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

def render_overview_tab(kpis: Dict[str, float]):
    """Render the Overview tab with enhanced KPI display"""
    st.markdown("### 📊 Performance Overview")
    st.markdown("Real-time key performance indicators for your Shell Eco-marathon vehicle")
    
    # Create enhanced KPI layout
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
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            return fig
    except Exception as e:
        st.error(f"Error creating {title}: {e}")
        return None

# Chart creation functions (same as original, but with enhanced styling)
def create_speed_chart(df: pd.DataFrame):
    """Generate a line chart showing vehicle speed over time"""
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
    return fig

def create_power_chart(df: pd.DataFrame):
    """Generate a subplot chart displaying voltage, current, and power"""
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
    """Generate a subplot chart for IMU data"""
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
    
    # Gyroscope data
    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                      line=dict(color=colors_gyro[i], width=2)), row=1, col=1,
        )
    
    # Accelerometer data
    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                      line=dict(color=colors_accel[i], width=2)), row=2, col=1,
        )
    
    fig.update_layout(height=600, title_text="🎮 IMU Sensor Data Analysis")
    return fig

def create_imu_chart_2(df: pd.DataFrame):
    """Generate detailed IMU chart with individual subplots"""
    if df.empty or not all(col in df.columns for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("🌀 Gyro X", "🌀 Gyro Y", "🌀 Gyro Z", "📊 Accel X", "📊 Accel Y", "📊 Accel Z"),
        vertical_spacing=0.3, horizontal_spacing=0.1,
    )
    
    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]
    
    # Gyroscope data
    for i, (axis, color) in enumerate(zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}",
                      line=dict(color=color, width=2), showlegend=False), row=1, col=i+1,
        )
    
    # Accelerometer data
    for i, (axis, color) in enumerate(zip(["accel_x", "accel_y", "accel_z"], accel_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}",
                      line=dict(color=color, width=2), showlegend=False), row=2, col=i+1,
        )
    
    fig.update_layout(height=600, title_text="🎮 Detailed IMU Sensor Analysis")
    return fig

def create_efficiency_chart(df: pd.DataFrame):
    """Generate efficiency analysis scatter plot"""
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
    return fig

def create_gps_map(df: pd.DataFrame):
    """Generate GPS tracking map"""
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
    """Create a customizable chart based on user-defined configurations"""
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
    
    if not y_col or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
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
                    text="Need at least 2 numeric columns for heatmap",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )

def render_dynamic_charts_section(df: pd.DataFrame):
    """Render the custom charts section (kept from original)"""
    st.markdown("### 📈 Custom Chart Builder")
    
    # Instructions
    st.markdown("""
    <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: var(--primary-color); margin-bottom: 0.5rem;">🎯 Create Custom Charts</h4>
        <p style="color: var(--text-secondary); margin-bottom: 0;">
            Click <strong>"Add Chart"</strong> to create custom visualizations with your preferred variables and chart types.
        </p>
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
    
    # Chart controls
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
    
    # Display charts
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
                            index=["line", "scatter", "bar", "histogram", "heatmap"].index(chart_config.get("chart_type", "line")),
                            key=f"type_{chart_config['id']}"
                        )
                        if new_type != chart_config.get("chart_type"):
                            st.session_state.dynamic_charts[i]["chart_type"] = new_type
                    
                    with col3:
                        if chart_config.get("chart_type", "line") not in ["histogram", "heatmap"]:
                            x_options = ["timestamp"] + available_columns if "timestamp" in df.columns else available_columns
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
                    
                    # Display chart
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

async def load_data_sources(subscriber: EnhancedTelemetrySubscriber, data_source: DataSourceType, selected_session: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data from multiple sources with triangulation"""
    try:
        data_sources = []
        source_info = {"realtime": 0, "ably_history": 0, "supabase": 0, "total": 0}
        
        if data_source == DataSourceType.REALTIME_PLUS_SESSION:
            # Get real-time messages
            realtime_messages = subscriber.get_messages()
            if realtime_messages:
                realtime_df = pd.DataFrame(realtime_messages)
                if not realtime_df.empty:
                    data_sources.append(realtime_df)
                    source_info["realtime"] = len(realtime_messages)
            
            # Get Ably history
            ably_history = await subscriber.get_ably_history(limit=ABLY_HISTORY_LIMIT)
            if ably_history:
                ably_df = pd.DataFrame(ably_history)
                if not ably_df.empty:
                    data_sources.append(ably_df)
                    source_info["ably_history"] = len(ably_history)
            
            # Get Supabase data for current session
            supabase_data = subscriber.get_supabase_data()
            if supabase_data:
                supabase_df = pd.DataFrame(supabase_data)
                if not supabase_df.empty:
                    data_sources.append(supabase_df)
                    source_info["supabase"] = len(supabase_data)
        
        elif data_source == DataSourceType.HISTORICAL:
            # Get historical data from Supabase
            if selected_session:
                supabase_data = subscriber.get_supabase_data(session_id=selected_session)
                if supabase_data:
                    supabase_df = pd.DataFrame(supabase_data)
                    if not supabase_df.empty:
                        data_sources.append(supabase_df)
                        source_info["supabase"] = len(supabase_data)
        
        # Triangulate and merge data sources
        if data_sources:
            merged_df = st.session_state.data_triangulator.merge_data_sources(data_sources)
            
            # Ensure timestamp is properly formatted
            if 'timestamp' in merged_df.columns:
                merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
                merged_df = merged_df.sort_values('timestamp')
            
            # Apply data point limit
            if len(merged_df) > MAX_DATAPOINTS:
                merged_df = merged_df.tail(MAX_DATAPOINTS)
            
            source_info["total"] = len(merged_df)
            return merged_df, source_info
        
        return pd.DataFrame(), source_info
        
    except Exception as e:
        st.error(f"Error loading data sources: {e}")
        return pd.DataFrame(), {"error": str(e)}

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<div class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</div>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Connection & Data Control")
        
        # Data source selection
        st.markdown('<div class="data-source-selector">', unsafe_allow_html=True)
        st.subheader("📊 Data Source")
        
        data_source_options = {
            "Real-time + Session Data": DataSourceType.REALTIME_PLUS_SESSION,
            "Historical Data": DataSourceType.HISTORICAL
        }
        
        selected_source = st.selectbox(
            "Choose data source:",
            options=list(data_source_options.keys()),
            index=0 if st.session_state.current_data_source == DataSourceType.REALTIME_PLUS_SESSION else 1
        )
        
        st.session_state.current_data_source = data_source_options[selected_source]
        st.session_state.is_historical_mode = (st.session_state.current_data_source == DataSourceType.HISTORICAL)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Connection controls
        if st.session_state.current_data_source == DataSourceType.REALTIME_PLUS_SESSION:
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
        
        # Session selection for historical data
        if st.session_state.current_data_source == DataSourceType.HISTORICAL:
            st.subheader("📋 Session Selection")
            
            if st.button("🔄 Refresh Sessions", use_container_width=True):
                if not st.session_state.subscriber:
                    st.session_state.subscriber = EnhancedTelemetrySubscriber()
                
                st.session_state.available_sessions = st.session_state.subscriber.get_supabase_sessions()
                st.rerun()
            
            # Load sessions if not already loaded
            if not st.session_state.available_sessions and st.session_state.subscriber:
                st.session_state.available_sessions = st.session_state.subscriber.get_supabase_sessions()
            
            if st.session_state.available_sessions:
                session_options = []
                for session in st.session_state.available_sessions:
                    label = f"{'🟢 ' if session.is_current else '🔵 '}{session.session_id[:8]}... ({session.record_count:,} records)"
                    session_options.append(label)
                
                selected_session_idx = st.selectbox(
                    "Select session:",
                    options=range(len(session_options)),
                    format_func=lambda x: session_options[x],
                    index=0 if st.session_state.selected_session is None else 
                          next((i for i, s in enumerate(st.session_state.available_sessions) 
                               if s.session_id == st.session_state.selected_session), 0)
                )
                
                st.session_state.selected_session = st.session_state.available_sessions[selected_session_idx].session_id
                
                # Display selected session info
                selected_session_info = st.session_state.available_sessions[selected_session_idx]
                render_session_card(selected_session_info)
            else:
                st.warning("No sessions available. Please ensure the bridge is running and has collected data.")
        
        # Connection status
        if st.session_state.subscriber:
            stats = st.session_state.subscriber.get_stats()
            
            if st.session_state.current_data_source == DataSourceType.REALTIME_PLUS_SESSION:
                if st.session_state.subscriber.is_connected:
                    st.markdown(
                        '<div class="status-indicator status-connected">✅ Connected</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="status-indicator status-disconnected">❌ Disconnected</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="status-indicator status-historical">📚 Historical Mode</div>',
                    unsafe_allow_html=True
                )
            
            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📨 Messages", stats["messages_received"])
                st.metric("📚 Ably History", stats["ably_history_retrieved"])
            with col2:
                st.metric("💾 Supabase", stats["supabase_records_retrieved"])
                st.metric("❌ Errors", stats["errors"])
            
            if stats["last_error"]:
                st.error(f"⚠️ {stats['last_error'][:50]}...")
        
        # Auto-refresh for real-time mode
        if st.session_state.current_data_source == DataSourceType.REALTIME_PLUS_SESSION:
            st.divider()
            st.subheader("⚙️ Settings")
            
            new_auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
            if new_auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = new_auto_refresh
            
            if st.session_state.auto_refresh:
                refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)
        
        st.info(f"📡 Channel: {CHANNEL_NAME}")
        st.info(f"💾 Database: {SUPABASE_TABLE_NAME}")
    
    # Main content area
    if st.session_state.current_data_source == DataSourceType.HISTORICAL and st.session_state.is_historical_mode:
        st.markdown(
            '<div class="historical-warning">📚 Viewing Historical Data - Auto-refresh is disabled</div>',
            unsafe_allow_html=True
        )
    
    # Load data
    if not st.session_state.subscriber:
        st.session_state.subscriber = EnhancedTelemetrySubscriber()
    
    # Load data asynchronously
    async def load_data():
        return await load_data_sources(
            st.session_state.subscriber,
            st.session_state.current_data_source,
            st.session_state.selected_session
        )
    
    # Create event loop and load data
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df, source_info = loop.run_until_complete(load_data())
        loop.close()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()
        source_info = {"error": str(e)}
    
    # Update session state
    st.session_state.telemetry_data = df
    st.session_state.last_update = datetime.now()
    
    # Display data status
    if df.empty:
        if st.session_state.current_data_source == DataSourceType.HISTORICAL:
            st.warning("⏳ No historical data available for the selected session.")
        else:
            st.warning("⏳ Waiting for telemetry data...")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Getting Started:**\n"
                "1. Ensure m1.py bridge is running\n"
                "2. Click 'Connect' to start receiving data\n"
                "3. Select sessions for historical analysis"
            )
        with col2:
            with st.expander("🔍 Debug Information"):
                st.json({
                    "Data Source": st.session_state.current_data_source.value,
                    "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                    "Selected Session": st.session_state.selected_session,
                    "Available Sessions": len(st.session_state.available_sessions),
                    "Source Info": source_info,
                })
        return
    
    # Display data summary
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points loaded")
    with col2:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col3:
        if "total" in source_info:
            st.success(f"📈 {source_info['total']:,}")
    
    # Data source breakdown
    if source_info and any(k in source_info for k in ["realtime", "ably_history", "supabase"]):
        with st.expander("📊 Data Source Breakdown"):
            source_col1, source_col2, source_col3 = st.columns(3)
            with source_col1:
                st.metric("🔴 Real-time", source_info.get("realtime", 0))
            with source_col2:
                st.metric("📚 Ably History", source_info.get("ably_history", 0))
            with source_col3:
                st.metric("💾 Supabase", source_info.get("supabase", 0))
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Tabs
    st.subheader("📈 Dashboard")
    
    tab_names = [
        "📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", 
        "🎮 IMU Detail", "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data"
    ]
    tabs = st.tabs(tab_names)
    
    # Tab content
    with tabs[0]:  # Overview
        render_overview_tab(kpis)
    
    with tabs[1]:  # Speed
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_speed_chart, "Speed Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Power
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_power_chart, "Power Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:  # IMU
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart, "IMU Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:  # IMU Detail
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart_2, "IMU Detail Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:  # Efficiency
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_efficiency_chart, "Efficiency Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[6]:  # GPS
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[7]:  # Custom
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)
    
    with tabs[8]:  # Data
        render_kpi_header(kpis)
        
        st.subheader("📃 Raw Telemetry Data")
        st.warning("ℹ️ Only the **last 1000 datapoints** are displayed below. Download the CSV for the complete dataset.")
        
        display_df = df.tail(1000)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=csv,
            file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    # Auto-refresh for real-time mode
    if (st.session_state.auto_refresh and 
        st.session_state.current_data_source == DataSourceType.REALTIME_PLUS_SESSION and
        st.session_state.subscriber and st.session_state.subscriber.is_connected):
        
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 1rem;">
        <p><strong>Shell Eco-marathon Telemetry Dashboard v2.0</strong> | Enhanced Multi-Source Data Visualization</p>
        <p>🚗 Real-time + Historical Analysis | 💾 Supabase Integration | 🔄 Data Triangulation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
