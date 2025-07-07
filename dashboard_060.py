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

# Disables tracemalloc warnings that can appear in Streamlit environments.
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Handles Ably library import with error checking.
try:
    from ably import AblyRealtime, AblyRest
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("❌ Ably library not available. Please install: pip install ably")
    st.stop()

# Supabase import (optional)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# Function to set up terminal logging.
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger("TelemetrySubscriber")

    # Prevents adding handlers multiple times on Streamlit reruns.
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

# Global configuration variables for Ably and data limits.
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATAPOINTS = 10000  # Reduced since we're getting persistent data
HISTORY_REFRESH_INTERVAL = 30  # Refresh history every 30 seconds

# Supabase configuration (you'll need to set these up)
SUPABASE_URL = "your-supabase-url"  # Replace with your Supabase URL
SUPABASE_KEY = "your-supabase-key"  # Replace with your Supabase anon key

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

# Enhanced CSS with better dark/light theme compatibility
st.markdown(
    """
<style>
    /* Theme-aware color variables */
    :root {
        --primary-color: #2E86AB;
        --primary-light: #A23B72;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --error-color: #E74C3C;
        --text-primary: #2C3E50;
        --text-secondary: #7F8C8D;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F8F9FA;
        --border-color: #E1E8ED;
        --card-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Dark theme overrides */
    [data-theme="dark"] {
        --primary-color: #5DADE2;
        --primary-light: #BB7FD9;
        --success-color: #58D68D;
        --warning-color: #F8C471;
        --error-color: #EC7063;
        --text-primary: #FAFAFA;
        --text-secondary: #BDC3C7;
        --bg-primary: #2C3E50;
        --bg-secondary: #34495E;
        --border-color: #566573;
        --card-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Auto-detect system theme */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #5DADE2;
            --primary-light: #BB7FD9;
            --success-color: #58D68D;
            --warning-color: #F8C471;
            --error-color: #EC7063;
            --text-primary: #FAFAFA;
            --text-secondary: #BDC3C7;
            --bg-primary: #2C3E50;
            --bg-secondary: #34495E;
            --border-color: #566573;
            --card-shadow: 0 2px 8px rgba(0,0,0,0.3);
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
        box-shadow: var(--card-shadow);
    }

    .status-connected {
        background: linear-gradient(135deg, var(--success-color), #A8E6CF);
        color: white;
        border: 2px solid var(--success-color);
    }

    .status-disconnected {
        background: linear-gradient(135deg, var(--error-color), #FFABAB);
        color: white;
        border: 2px solid var(--error-color);
    }

    .status-connecting {
        background: linear-gradient(135deg, var(--warning-color), #FFD93D);
        color: white;
        border: 2px solid var(--warning-color);
    }

    .status-loading-history {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        color: white;
        border: 2px solid var(--primary-color);
    }

    .history-status {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: var(--card-shadow);
    }

    .supabase-section {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
    }

    .supabase-title {
        color: var(--primary-color);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .instructions-container {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 6px solid var(--primary-color);
        box-shadow: var(--card-shadow);
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
        box-shadow: var(--card-shadow);
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

    /* Improved button styling */
    .stButton > button {
        border-radius: 8px;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--card-shadow);
    }

    .stButton > button:hover {
        background: var(--primary-light);
        border-color: var(--primary-light);
        transform: translateY(-1px);
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
    """Enhanced telemetry subscriber with persistent message history support."""

    def __init__(self):
        self.ably_client = None
        self.ably_rest_client = None
        self.channel = None
        self.rest_channel = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.connection_thread = None
        self.history_thread = None
        self.stats = {
            "messages_received": 0,
            "history_messages_loaded": 0,
            "last_message_time": None,
            "last_history_refresh": None,
            "connection_attempts": 0,
            "errors": 0,
            "last_error": None,
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self._history_data = []
        self._last_history_refresh = None

        # Initialize logger for this class.
        self.logger = logging.getLogger("TelemetrySubscriber")

    def connect(self) -> bool:
        """Connect to Ably and start receiving messages and history"""
        try:
            with self._lock:
                self.stats["connection_attempts"] += 1

            self.logger.info("🔌 Starting connection to Ably...")

            # Stop any existing connection before establishing a new one.
            if self._should_run:
                self.disconnect()

            # Clear the stop event and set the running flag for a new connection.
            self._stop_event.clear()
            self._should_run = True

            # Start the connection handling in a separate thread.
            self.connection_thread = threading.Thread(
                target=self._connection_worker, daemon=True
            )
            self.connection_thread.start()

            # Start the history refresh worker
            self.history_thread = threading.Thread(
                target=self._history_worker, daemon=True
            )
            self.history_thread.start()

            # Pause briefly to allow the connection thread to initiate.
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

            # Create a new event loop specific to this thread.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Execute the asynchronous connection handling coroutine.
            loop.run_until_complete(self._async_connection_handler())

        except Exception as e:
            self.logger.error(f"💥 Connection worker error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False
        finally:
            self.logger.info("🛑 Connection worker ended")

    def _history_worker(self):
        """Worker thread to handle periodic history refresh"""
        try:
            self.logger.info("📚 History worker starting...")
            
            while self._should_run and not self._stop_event.is_set():
                try:
                    # Load history on first run or every HISTORY_REFRESH_INTERVAL
                    if (self._last_history_refresh is None or 
                        time.time() - self._last_history_refresh > HISTORY_REFRESH_INTERVAL):
                        self._load_history()
                        self._last_history_refresh = time.time()
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"❌ History worker error: {e}")
                    with self._lock:
                        self.stats["errors"] += 1
                        self.stats["last_error"] = f"History error: {e}"
                    time.sleep(10)  # Wait longer on error
            
            self.logger.info("📚 History worker ended")
            
        except Exception as e:
            self.logger.error(f"💥 History worker fatal error: {e}")

    def _load_history(self):
        """Load message history from Ably"""
        try:
            self.logger.info("📚 Loading message history...")
            
            # Create REST client if not exists
            if not self.ably_rest_client:
                self.ably_rest_client = AblyRest(ABLY_API_KEY)
                self.rest_channel = self.ably_rest_client.channels.get(CHANNEL_NAME)
            
            # Get 24 hours of history (adjust as needed)
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours ago
            
            # Fetch history with pagination
            history_messages = []
            
            # Get history from the REST API
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Use the REST API to get history
                history_result = loop.run_until_complete(
                    self.rest_channel.history({
                        'start': start_time,
                        'end': end_time,
                        'limit': 1000,
                        'direction': 'backwards'
                    })
                )
                
                # Process the history messages
                for message in history_result.items:
                    if message.name == 'telemetry_update':
                        try:
                            data = message.data
                            if isinstance(data, str):
                                data = json.loads(data)
                            
                            # Add timestamp from message if not present
                            if 'timestamp' not in data:
                                data['timestamp'] = datetime.fromtimestamp(message.timestamp / 1000).isoformat()
                            
                            # Mark as historical data
                            data['data_source'] = data.get('data_source', 'HISTORICAL')
                            history_messages.append(data)
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error processing history message: {e}")
                            continue
                
                # Update history data
                with self._lock:
                    self._history_data = history_messages
                    self.stats["history_messages_loaded"] = len(history_messages)
                    self.stats["last_history_refresh"] = datetime.now()
                
                self.logger.info(f"📚 Loaded {len(history_messages)} historical messages")
                
            finally:
                loop.close()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load history: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"History load error: {e}"

    def force_history_refresh(self):
        """Force an immediate history refresh"""
        try:
            self.logger.info("🔄 Forcing history refresh...")
            self._last_history_refresh = None  # Reset to force immediate refresh
            
            # Trigger history loading in a separate thread to avoid blocking
            history_thread = threading.Thread(target=self._load_history, daemon=True)
            history_thread.start()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to force history refresh: {e}")

    def get_history_data(self) -> List[Dict[str, Any]]:
        """Get the loaded history data"""
        with self._lock:
            return self._history_data.copy()

    async def _async_connection_handler(self):
        """Handle Ably connection asynchronously"""
        try:
            self.logger.info("🔑 Creating Ably client...")

            # Instantiate the Ably Realtime client using the API key.
            self.ably_client = AblyRealtime(ABLY_API_KEY)

            # Define callback functions for various connection state changes.
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

            # Attach the defined handlers to the Ably connection events.
            self.ably_client.connection.on("connected", on_connected)
            self.ably_client.connection.on("disconnected", on_disconnected)
            self.ably_client.connection.on("failed", on_failed)
            self.ably_client.connection.on("suspended", on_disconnected)

            # Wait until the Ably client successfully connects.
            self.logger.info("⏳ Waiting for connection...")
            await self.ably_client.connection.once_async("connected")

            # Retrieve the specified Ably channel.
            self.logger.info(f"📡 Getting channel: {CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(CHANNEL_NAME)

            # Subscribe to messages with the "telemetry_update" name on the channel.
            self.logger.info("📨 Subscribing to messages...")
            await self.channel.subscribe("telemetry_update", self._on_message_received)

            self.logger.info("✅ Successfully subscribed to messages!")

            # Continuously run a loop to keep the connection alive while the application is active.
            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)

                # Check the current state of the Ably connection.
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

            # Retrieve the data payload from the received message.
            data = message.data

            # Parse the message data as JSON if it is a string.
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON decode error: {e}")
                    with self._lock:
                        self.stats["errors"] += 1
                        self.stats["last_error"] = f"JSON decode error: {e}"
                    return

            # Validate that the received data is a dictionary.
            if not isinstance(data, dict):
                self.logger.warning(f"⚠️ Invalid data type: {type(data)}")
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"Invalid data type: {type(data)}"
                return

            self.logger.debug(f"📊 Data keys: {list(data.keys())}")

            # Add the processed message data to an internal queue.
            with self._lock:
                # Reduced queue size since we're getting persistent data
                if self.message_queue.qsize() > 50:
                    try:
                        # Remove messages from the queue until a manageable size is reached.
                        while self.message_queue.qsize() > 25:
                            self.message_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Add the new message to the queue and update statistics.
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

            # Signal the connection loop and running flag to stop.
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False

            # Attempt to close the Ably client connection gracefully.
            if self.ably_client:
                try:
                    self.ably_client.close()
                    self.logger.info("✅ Ably realtime connection closed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably realtime: {e}")

            if self.ably_rest_client:
                try:
                    self.ably_rest_client.close()
                    self.logger.info("✅ Ably REST connection closed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably REST: {e}")

            # Wait for the connection thread to terminate.
            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=5)
                if self.connection_thread.is_alive():
                    self.logger.warning("⚠️ Connection thread did not stop gracefully")

            # Wait for the history thread to terminate.
            if self.history_thread and self.history_thread.is_alive():
                self.history_thread.join(timeout=5)
                if self.history_thread.is_alive():
                    self.logger.warning("⚠️ History thread did not stop gracefully")

            self.logger.info("🔚 Disconnection complete")

        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Disconnect error: {e}"
        finally:
            self.ably_client = None
            self.ably_rest_client = None
            self.channel = None
            self.rest_channel = None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self._lock:
            return self.stats.copy()


class SupabaseManager:
    """Handles Supabase database operations for historical session data"""
    
    def __init__(self):
        self.client = None
        self.is_connected = False
        
        if SUPABASE_AVAILABLE and SUPABASE_URL != "your-supabase-url":
            try:
                self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
                self.is_connected = True
            except Exception as e:
                logging.error(f"Failed to connect to Supabase: {e}")
                self.is_connected = False
    
    def get_available_sessions(self) -> List[Dict[str, Any]]:
        """Get list of available telemetry sessions"""
        if not self.is_connected:
            return []
        
        try:
            # This assumes you have a 'sessions' table with session metadata
            response = self.client.table('sessions').select('*').order('created_at', desc=True).execute()
            return response.data
        except Exception as e:
            logging.error(f"Failed to get sessions: {e}")
            return []
    
    def get_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Get telemetry data for a specific session"""
        if not self.is_connected:
            return []
        
        try:
            # This assumes you have a 'telemetry_data' table with session data
            response = self.client.table('telemetry_data').select('*').eq('session_id', session_id).order('timestamp').execute()
            return response.data
        except Exception as e:
            logging.error(f"Failed to get session data: {e}")
            return []


def initialize_session_state():
    """Initialize Streamlit session state variables with default values if they don't exist."""
    defaults = {
        "subscriber": None,
        "supabase_manager": SupabaseManager(),
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "active_tab": 0,
        "is_auto_refresh": False,
        "scroll_position": 0,
        "selected_session": None,
        "last_history_load": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators from the telemetry DataFrame, with error handling for missing data."""
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
        # Ensure specified columns are numeric, coercing errors to NaN and filling NaNs with 0.
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

        # Calculate individual KPIs, ensuring non-negative values and handling potential division by zero.
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

        # Calculate efficiency, protecting against division by zero.
        if kpis["total_energy_mj"] > 0:
            kpis["efficiency_km_per_mj"] = (
                kpis["total_distance_km"] / kpis["total_energy_mj"]
            )

        if "total_acceleration" in df.columns:
            accel_data = df["total_acceleration"].dropna()
            if not accel_data.empty:
                kpis["max_acceleration"] = max(0, accel_data.max())

        # Calculate the average magnitude of gyroscope data.
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
    """Render a compact performance dashboard at the top of a tab using Streamlit columns."""
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
    """Render the Overview tab with enhanced KPI display using Streamlit native components."""
    st.markdown("### 📊 Performance Overview")
    st.markdown(
        "Real-time key performance indicators for your Shell Eco-marathon vehicle"
    )

    # Create KPI layout using Streamlit columns.
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
    """Render connection status and statistics in the sidebar."""
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

    # Display connection statistics in a compact, two-column layout.
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

    # Display history status
    st.sidebar.markdown("### 📚 History Status")
    
    if stats.get("history_messages_loaded", 0) > 0:
        st.sidebar.markdown(
            f'<div class="history-status">📚 History: {stats["history_messages_loaded"]} messages loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="history-status">📚 History: No data loaded</div>',
            unsafe_allow_html=True,
        )
    
    if stats.get("last_history_refresh"):
        last_refresh = stats["last_history_refresh"]
        time_since_refresh = (datetime.now() - last_refresh).total_seconds()
        st.sidebar.info(f"⏰ Last refresh: {time_since_refresh:.0f}s ago")


def render_supabase_section():
    """Render the Supabase historical sessions section in the sidebar."""
    st.sidebar.markdown(
        """
        <div class="supabase-section">
            <div class="supabase-title">
                🗄️ Historical Sessions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not SUPABASE_AVAILABLE:
        st.sidebar.warning("⚠️ Supabase library not installed")
        st.sidebar.code("pip install supabase", language="bash")
        return

    supabase_manager = st.session_state.supabase_manager

    if not supabase_manager.is_connected:
        st.sidebar.warning("⚠️ Supabase not configured")
        st.sidebar.info("Update SUPABASE_URL and SUPABASE_KEY in the code")
        return

    # Get available sessions
    try:
        sessions = supabase_manager.get_available_sessions()
        
        if not sessions:
            st.sidebar.info("No historical sessions available")
            return
        
        # Create session selection dropdown
        session_options = ["Select a session..."] + [
            f"{session['name']} ({session['created_at'][:10]})"
            for session in sessions
        ]
        
        selected_session_idx = st.sidebar.selectbox(
            "Select Historical Session",
            range(len(session_options)),
            format_func=lambda x: session_options[x],
            key="session_selector"
        )
        
        if selected_session_idx > 0:
            selected_session = sessions[selected_session_idx - 1]
            st.session_state.selected_session = selected_session
            
            # Show session details
            st.sidebar.info(f"📊 Session: {selected_session['name']}")
            st.sidebar.info(f"📅 Date: {selected_session['created_at'][:10]}")
            
            # Load session data button
            if st.sidebar.button("📥 Load Session Data", key="load_session"):
                with st.spinner("Loading session data..."):
                    session_data = supabase_manager.get_session_data(selected_session['id'])
                    
                    if session_data:
                        # Convert to DataFrame and update session state
                        df = pd.DataFrame(session_data)
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        st.session_state.telemetry_data = df
                        st.session_state.last_update = datetime.now()
                        st.success(f"✅ Loaded {len(session_data)} data points")
                        st.rerun()
                    else:
                        st.error("❌ No data found for this session")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading sessions: {e}")


def create_optimized_chart(df: pd.DataFrame, chart_func, title: str):
    """Create an optimized Plotly chart by applying consistent styling."""
    try:
        fig = chart_func(df)
        if fig:
            # Apply consistent theme-aware styling to the chart layout.
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="var(--text-primary)"),
                title=dict(font=dict(size=16, color="var(--primary-color)")),
                margin=dict(l=40, r=40, t=60, b=40),
                height=400,
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            )
            return fig
    except Exception as e:
        st.error(f"Error creating {title}: {e}")
        return None


def create_speed_chart(df: pd.DataFrame):
    """Generate a line chart showing vehicle speed over time."""
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
        color_discrete_sequence=["#2E86AB"],
    )
    return fig


def create_power_chart(df: pd.DataFrame):
    """Generate a subplot chart displaying voltage, current, and power output over time."""
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
            line=dict(color="#27AE60", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["current_a"],
            name="Current (A)",
            line=dict(color="#E74C3C", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["power_w"],
            name="Power (W)",
            line=dict(color="#F39C12", width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=500, title_text="⚡ Electrical System Performance")
    return fig


def create_imu_chart(df: pd.DataFrame):
    """Generate a subplot chart for IMU data, displaying gyroscope and accelerometer readings."""
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

    # Add gyroscope data traces with distinct colors.
    colors_gyro = ["#E74C3C", "#27AE60", "#2E86AB"]
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

    # Add accelerometer data traces with distinct colors.
    colors_accel = ["#F39C12", "#A23B72", "#566573"]
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
    """Generate a detailed IMU chart with individual subplots for each gyroscope and accelerometer axis."""
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

    # Define color schemes for gyroscope and accelerometer plots.
    gyro_colors = ["#E74C3C", "#27AE60", "#2E86AB"]
    accel_colors = ["#F39C12", "#A23B72", "#566573"]

    # Add gyroscope data traces to their respective subplots.
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

    # Add accelerometer data traces to their respective subplots.
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
    """Generate a scatter plot for efficiency analysis, showing speed vs. power consumption."""
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
        color_continuous_scale="plasma",
    )
    return fig


def create_gps_map(df: pd.DataFrame):
    """Generate a scatter map to display vehicle GPS tracking and performance metrics."""
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    # Filter the DataFrame to include only rows with valid latitude and longitude.
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

    # Calculate the center point for the map view.
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
        zoom=15,  # Set the initial zoom level for the map.
        center=center_point,
        color_continuous_scale="plasma",
    )

    return fig


def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Retrieve a list of numeric columns from the DataFrame suitable for plotting."""
    if df.empty:
        return []

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create a customizable chart based on user-defined configurations, including heatmap support."""
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
                color_discrete_sequence=["#2E86AB"],
            )
        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#F39C12"],
            )
        elif chart_type == "bar":
            # Select the last 20 data points for the bar chart.
            recent_df = df.tail(20)
            fig = px.bar(
                recent_df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#27AE60"],
            )
        elif chart_type == "histogram":
            fig = px.histogram(
                df,
                x=y_col,
                title=f"Distribution of {y_col}",
                color_discrete_sequence=["#E74C3C"],
            )
        elif chart_type == "heatmap":
            # Create a heatmap for correlation analysis of numeric columns.
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
            # Default to a line chart if the type is not recognized.
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=["#2E86AB"],
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
    """Render the section for creating and displaying dynamic, user-configured charts."""

    st.session_state.is_auto_refresh = True

    # Display an enhanced instructions section using custom HTML styling.
    st.markdown(
        """
    <div class="instructions-container">
        <div class="instructions-title">
            🎯 Create Custom Charts
        </div>
        <div class="instructions-content">
            <p>Click <strong>"Add Chart"</strong> to create custom visualizations with your preferred variables and chart types.</p>
            <p><strong>Note:</strong> Chart rendering is optimized for better performance with persistent data loading.</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display information about different chart types in a grid layout.
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

    # Display controls for adding and managing dynamic charts.
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

    # Iterate through and display each dynamically configured chart.
    if st.session_state.dynamic_charts:
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            try:
                with st.container(border=True):
                    # Arrange chart configuration controls in a compact row.
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

                    # Display the dynamically created chart.
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
    """Main dashboard function, managing UI elements, data ingestion, and chart rendering."""
    # Render a sticky header at the top of the page.
    st.markdown(
        '<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>',
        unsafe_allow_html=True,
    )

    initialize_session_state()

    # Render sidebar elements for connection control and settings.
    with st.sidebar:
        st.header("🔧 Connection Control")

        # Arrange connection buttons in a single row.
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

        # Force history refresh button
        if st.button("🔄 Refresh History", use_container_width=True, help="Force reload of persistent message history"):
            if st.session_state.subscriber:
                st.session_state.subscriber.force_history_refresh()
                st.info("🔄 History refresh triggered")
            else:
                st.warning("⚠️ Not connected - connect first")

        # Display current connection status and statistics.
        stats = (
            st.session_state.subscriber.get_stats()
            if st.session_state.subscriber
            else {
                "messages_received": 0,
                "history_messages_loaded": 0,
                "connection_attempts": 0,
                "errors": 0,
                "last_message_time": None,
                "last_history_refresh": None,
                "last_error": None,
            }
        )

        render_connection_status(st.session_state.subscriber, stats)

        if stats["last_error"]:
            st.error(f"⚠️ {stats['last_error'][:40]}...")

        st.divider()

        # Display auto-refresh settings.
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
        st.info(f"🔄 History refresh: {HISTORY_REFRESH_INTERVAL}s")

        st.divider()

        # Render Supabase historical sessions section
        render_supabase_section()

    # Combine persistent history with new real-time messages
    combined_messages = []
    new_messages_count = 0

    # Get persistent history data
    if st.session_state.subscriber:
        history_data = st.session_state.subscriber.get_history_data()
        combined_messages.extend(history_data)

        # Get new real-time messages
        new_messages = st.session_state.subscriber.get_messages()
        if new_messages:
            new_messages_count = len(new_messages)
            combined_messages.extend(new_messages)

    # Process the combined messages
    if combined_messages:
        combined_df = pd.DataFrame(combined_messages)

        if "timestamp" in combined_df.columns:
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

        # Remove duplicates based on timestamp and message_id if available
        if "message_id" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["timestamp", "message_id"])
        else:
            combined_df = combined_df.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp")

        # Limit to maximum datapoints
        if len(combined_df) > MAX_DATAPOINTS:
            combined_df = combined_df.tail(MAX_DATAPOINTS)

        # Update session state
        st.session_state.telemetry_data = combined_df
        st.session_state.last_update = datetime.now()

    df = st.session_state.telemetry_data.copy()

    # Display an empty state message and debug information if no data is available.
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
                st.json(
                    {
                        "Connected": st.session_state.subscriber.is_connected
                        if st.session_state.subscriber
                        else False,
                        "Live Messages": stats["messages_received"],
                        "History Messages": stats["history_messages_loaded"],
                        "Errors": stats["errors"],
                        "Channel": CHANNEL_NAME,
                        "History Refresh": f"{HISTORY_REFRESH_INTERVAL}s",
                    }
                )
        return  # Halt rendering of the rest of the UI until data arrives.

    # Display a status row with data point count, last update time, and new message count.
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        st.info(f"📊 **{len(df):,}** data points collected")
    with col2:
        st.info(
            f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**"
        )
    with col3:
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")
    with col4:
        if stats.get("history_messages_loaded", 0) > 0:
            st.info(f"📚 {stats['history_messages_loaded']} hist.")

    # Calculate key performance indicators based on the current data.
    kpis = calculate_kpis(df)

    # Define and render the main dashboard tabs.
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

    # Render content for the Overview tab.
    with tabs[0]:
        render_overview_tab(kpis)

    # Render content for the Speed tab.
    with tabs[1]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_speed_chart, "Speed Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Render content for the Power tab.
    with tabs[2]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_power_chart, "Power Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Render content for the IMU tab.
    with tabs[3]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart, "IMU Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Render content for the IMU Detail tab.
    with tabs[4]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_imu_chart_2, "IMU Detail Chart")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Render content for the Efficiency tab.
    with tabs[5]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(
            df, create_efficiency_chart, "Efficiency Chart"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Render content for the GPS tab.
    with tabs[6]:
        render_kpi_header(kpis)
        fig = create_optimized_chart(df, create_gps_map, "GPS Map")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Render content for the Custom tab, allowing user-defined charts.
    with tabs[7]:
        render_kpi_header(kpis)
        render_dynamic_charts_section(df)

    # Render content for the Data tab, showing raw data and a download option.
    with tabs[8]:
        render_kpi_header(kpis)

        st.subheader("📃 Raw Telemetry Data")
        
        # Show data source breakdown
        if 'data_source' in df.columns:
            source_counts = df['data_source'].value_counts()
            st.info(f"Data sources: {dict(source_counts)}")
        
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

    # Implement auto-refresh functionality based on user settings.
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

    # Render the application footer.
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem;'>
            <p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | Enhanced with Persistent History</p>
            <p>🚗 Real-time + Historical Data | 📚 24-hour message persistence | 🗄️ Supabase integration</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
