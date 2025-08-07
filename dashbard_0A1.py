# fmt: off
# Note: Streamlit doesn't use Prettier; this code is wrapped to ~80 chars and
# formatted for readability.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta, timezone
import matplotlib.colors as mcolors
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

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# Third-party services (these imports are required by your app)
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
    st.error("❌ Supabase library not available. "
             "Please install: pip install supabase")
    st.stop()

# Silence tracemalloc warnings from external libs
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*tracemalloc.*"
)

# -----------------------------------------------------------------------------
# Configuration (Consider moving secrets to environment variables in production)
# -----------------------------------------------------------------------------
DASHBOARD_ABLY_API_KEY = (
    "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
)
DASHBOARD_CHANNEL_NAME = "telemetry-dashboard-channel"
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwi"
    "cm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0."
    "P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
)
SUPABASE_TABLE_NAME = "telemetry"

# Pagination constants
SUPABASE_MAX_ROWS_PER_REQUEST = 1000
MAX_DATAPOINTS_PER_SESSION = 1_000_000

# Page config
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

# -----------------------------------------------------------------------------
# Theme-aware Plotly
# -----------------------------------------------------------------------------
def get_theme_base() -> str:
    """
    Attempt to detect Streamlit theme (light/dark).
    Falls back to 'light' if unavailable.
    """
    try:
        # Streamlit 2025 added richer theming and st.context.theme
        # (See Streamlit 2025 release notes)
        # If unavailable, fallback to light.
        return getattr(st, "context").theme.get("base", "light")  # type: ignore
    except Exception:
        return "light"


def apply_plotly_template():
    """
    Set Plotly default template based on Streamlit theme base.
    """
    base = get_theme_base()
    if base.lower() == "dark":
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"


apply_plotly_template()

# -----------------------------------------------------------------------------
# Modern Theme-Aware CSS (Visual Refresh)
# -----------------------------------------------------------------------------
def get_theme_aware_css() -> str:
    """
    Minimalist modern theme:
      - OKLAB/OKLCH color system
      - Subtle glass surfaces with blur + soft shadows
      - Adaptive to light/dark via media queries
    """
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

:root {
  color-scheme: light dark;

  /* Accent */
  --accent-h: 210;
  --accent-s: 90%;
  --accent-l: 50%;
  --accent: hsl(var(--accent-h), var(--accent-s), var(--accent-l));
  --accent-10: hsla(var(--accent-h), var(--accent-s), var(--accent-l), 0.10);
  --accent-20: hsla(var(--accent-h), var(--accent-s), var(--accent-l), 0.20);

  /* Text & surfaces (adaptive via Canvas/CanvasText) */
  --bg: Canvas;
  --text: CanvasText;
  --text-muted: color-mix(in oklab, CanvasText 55%, Canvas);
  --text-subtle: color-mix(in oklab, CanvasText 40%, Canvas);

  /* Borders */
  --border: color-mix(in oklab, CanvasText 16%, Canvas);
  --border-strong: color-mix(in oklab, CanvasText 28%, Canvas);

  /* Glass surfaces */
  --glass: color-mix(in oklab, Canvas 75%, transparent);
  --glass-strong: color-mix(in oklab, Canvas 68%, transparent);
  --glass-border: color-mix(in oklab, CanvasText 18%, transparent);

  /* Shadows */
  --shadow-1: 0 6px 18px color-mix(in oklab, CanvasText 10%, transparent);
  --shadow-2: 0 10px 40px color-mix(in oklab, CanvasText 14%, transparent);
  --shadow-3: 0 18px 60px color-mix(in oklab, CanvasText 20%, transparent);

  /* Radii */
  --r-sm: 10px;
  --r-md: 14px;
  --r-lg: 18px;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  color: var(--text);
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 500px at 50% 110%,
      color-mix(in oklab, var(--accent) 12%, transparent) 0%,
      transparent 60%) no-repeat,
    var(--bg);
  background-attachment: fixed;
}

/* Header: translucent glass */
[data-testid="stHeader"] {
  background: var(--glass);
  backdrop-filter: blur(16px) saturate(115%);
  border-bottom: 1px solid var(--glass-border);
}

/* Cards */
.card, .chart-wrap, [data-testid="stExpander"], [data-testid="stDataFrame"] {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: var(--r-lg);
  backdrop-filter: blur(18px) saturate(120%);
  box-shadow: var(--shadow-1);
}

.card:hover, .chart-wrap:hover {
  box-shadow: var(--shadow-2);
  transform: translateY(-2px);
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}

/* Main Title */
.main-header {
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: 0.1px;
  text-align: center;
  color: var(--accent);
  margin-bottom: 0.6rem;
}

/* Status pills */
.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--glass-strong);
  border: 1px solid var(--border);
  border-radius: 9999px;
  font-weight: 700;
  color: var(--text);
  box-shadow: var(--shadow-1);
}

/* Buttons */
.stButton > button {
  border-radius: 12px !important;
  border: 1px solid var(--accent) !important;
  background: linear-gradient(180deg,
    color-mix(in oklab, var(--accent) 92%, white) 0%,
    var(--accent) 100%) !important;
  color: white !important;
  font-weight: 700 !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.stButton > button:hover {
  transform: translateY(-1px) scale(1.01);
  box-shadow: var(--shadow-1);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--border);
  gap: 6px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px 10px 0 0;
  color: var(--text-muted);
  font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text);
  background: color-mix(in oklab, var(--accent) 8%, transparent);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent);
  box-shadow: inset 0 -3px 0 0 var(--accent);
}

/* Grid for small gauges */
.widget-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(160px, 1fr));
  gap: 12px;
}
.gauge-container {
  border: 1px solid var(--glass-border);
  border-radius: var(--r-md);
  background: var(--glass);
  padding: 10px;
  transition: all 0.25s ease;
}
.gauge-container:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-1);
  background: color-mix(in oklab, var(--accent) 5%, var(--glass));
}

.gauge-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-subtle);
  margin-bottom: 4px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* Focus visible */
*:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 6px;
}

/* Sidebar glass */
.css-1d391kg, [data-testid="stSidebar"] {
  background: var(--glass-strong) !important;
  backdrop-filter: blur(18px) saturate(115%) !important;
  border-right: 1px solid var(--glass-border);
}
</style>
    """


# Apply modern CSS
st.markdown(get_theme_aware_css(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------------
def setup_terminal_logging():
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

# -----------------------------------------------------------------------------
# Telemetry Manager
# -----------------------------------------------------------------------------
class EnhancedTelemetryManager:
    """Telemetry manager with multi-source data and pagination."""

    def __init__(self):
        self.realtime_subscriber = None
        self.supabase_client: Optional[Client] = None
        self.is_connected = False
        self.message_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.connection_thread: Optional[threading.Thread] = None
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
                "sessions_paginated": 0,
            },
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self.logger = logging.getLogger("TelemetryDashboard")

    def connect_supabase(self) -> bool:
        """Connect to Supabase."""
        try:
            self.supabase_client = create_client(
                SUPABASE_URL, SUPABASE_API_KEY
            )
            self.logger.info("✅ Connected to Supabase")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Supabase: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return False

    def connect_realtime(self) -> bool:
        """Connect to Ably for realtime data."""
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
            self.realtime_subscriber.connection.on(
                "disconnected", on_disconnected
            )
            self.realtime_subscriber.connection.on("failed", on_failed)

            await self.realtime_subscriber.connection.once_async("connected")

            channel = self.realtime_subscriber.channels.get(
                DASHBOARD_CHANNEL_NAME
            )
            await channel.subscribe(
                "telemetry_update", self._on_message_received
            )

            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"💥 Async connection error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            self.is_connected = False

    def _on_message_received(self, message):
        """Handle incoming messages."""
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

                data["data_source"] = "realtime"
                self.message_queue.put(data)
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = datetime.now()
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)

    def get_realtime_messages(self) -> List[Dict[str, Any]]:
        """Return all queued realtime messages."""
        messages: List[Dict[str, Any]] = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    messages.append(self.message_queue.get_nowait())
                except queue.Empty:
                    break
        return messages

    def _paginated_fetch(
        self, session_id: str, data_source: str = "supabase_current"
    ) -> pd.DataFrame:
        """
        Fetch paginated rows from Supabase. Uses .range() pagination and .order().
        See Supabase Python docs for range/order usage.
        """
        try:
            if not self.supabase_client:
                self.logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()

            all_data: List[Dict[str, Any]] = []
            offset = 0
            total_fetched = 0
            request_count = 0
            max_requests = math.ceil(
                MAX_DATAPOINTS_PER_SESSION / SUPABASE_MAX_ROWS_PER_REQUEST
            )

            self.logger.info(
                f"🔄 Starting paginated fetch for session "
                f"{session_id[:8]}..."
            )

            while offset < MAX_DATAPOINTS_PER_SESSION:
                try:
                    range_end = offset + SUPABASE_MAX_ROWS_PER_REQUEST - 1
                    self.logger.info(
                        f"📄 Fetch page {request_count + 1}: "
                        f"rows {offset}-{range_end}"
                    )
                    response = (
                        self.supabase_client.table(SUPABASE_TABLE_NAME)
                        .select("*")
                        .eq("session_id", session_id)
                        .order("timestamp", desc=False)
                        .range(offset, range_end)
                        .execute()
                    )
                    request_count += 1

                    if not response.data:
                        self.logger.info(
                            f"✅ No more data found at offset {offset}"
                        )
                        break

                    batch_size = len(response.data)
                    all_data.extend(response.data)
                    total_fetched += batch_size

                    self.logger.info(
                        f"📊 Fetched {batch_size} rows "
                        f"(total: {total_fetched})"
                    )

                    if batch_size < SUPABASE_MAX_ROWS_PER_REQUEST:
                        self.logger.info("✅ Reached end of data")
                        break

                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(
                        f"❌ Error in pagination request {request_count}: {e}"
                    )
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    continue

            with self._lock:
                self.stats["pagination_stats"]["total_requests"] += \
                    request_count
                self.stats["pagination_stats"]["total_rows_fetched"] += \
                    total_fetched
                self.stats["pagination_stats"]["largest_session_size"] = max(
                    self.stats["pagination_stats"]["largest_session_size"],
                    total_fetched,
                )
                if request_count > 1:
                    self.stats["pagination_stats"]["sessions_paginated"] += 1

            if all_data:
                df = pd.DataFrame(all_data)
                df["data_source"] = data_source
                self.logger.info(
                    f"✅ Successfully fetched {len(df)} total rows for "
                    f"session {session_id[:8]}..."
                )
                return df
            else:
                self.logger.warning(f"⚠️ No data found for session {session_id}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(
                f"❌ Error in paginated fetch for session {session_id}: {e}"
            )
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return pd.DataFrame()

    def get_current_session_data(self, session_id: str) -> pd.DataFrame:
        self.logger.info(
            f"🔄 Fetching current session data for {session_id[:8]}..."
        )
        return self._paginated_fetch(session_id, "supabase_current")

    def get_historical_sessions(self) -> List[Dict[str, Any]]:
        """
        Return unique sessions list with start/end/duration/record_count.
        """
        try:
            if not self.supabase_client:
                self.logger.error("❌ Supabase client not initialized")
                return []

            self.logger.info("🔄 Fetching historical sessions list...")

            all_records: List[Dict[str, Any]] = []
            offset = 0

            while True:
                try:
                    range_end = offset + SUPABASE_MAX_ROWS_PER_REQUEST - 1
                    response = (
                        self.supabase_client.table(SUPABASE_TABLE_NAME)
                        .select("session_id, timestamp")
                        .order("timestamp", desc=True)
                        .range(offset, range_end)
                        .execute()
                    )
                    if not response.data:
                        break
                    all_records.extend(response.data)
                    if len(response.data) < SUPABASE_MAX_ROWS_PER_REQUEST:
                        break
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                except Exception as e:
                    self.logger.error(
                        f"❌ Error fetching session records at offset "
                        f"{offset}: {e}"
                    )
                    break

            if not all_records:
                self.logger.warning("⚠️ No session records found")
                return []

            sessions: Dict[str, Dict[str, Any]] = {}
            for record in all_records:
                session_id = record["session_id"]
                timestamp = record["timestamp"]
                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "start_time": timestamp,
                        "end_time": timestamp,
                        "record_count": 1,
                    }
                else:
                    sessions[session_id]["record_count"] += 1
                    if timestamp < sessions[session_id]["start_time"]:
                        sessions[session_id]["start_time"] = timestamp
                    if timestamp > sessions[session_id]["end_time"]:
                        sessions[session_id]["end_time"] = timestamp

            session_list: List[Dict[str, Any]] = []
            for s in sessions.values():
                try:
                    start_dt = datetime.fromisoformat(
                        s["start_time"].replace("Z", "+00:00")
                    )
                    end_dt = datetime.fromisoformat(
                        s["end_time"].replace("Z", "+00:00")
                    )
                    duration = end_dt - start_dt
                    session_list.append(
                        {
                            "session_id": s["session_id"],
                            "start_time": start_dt,
                            "end_time": end_dt,
                            "duration": duration,
                            "record_count": s["record_count"],
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"❌ Error processing session {s['session_id']}: {e}"
                    )

            sorted_sessions = sorted(
                session_list, key=lambda x: x["start_time"], reverse=True
            )
            self.logger.info(f"✅ Found {len(sorted_sessions)} unique sessions")
            return sorted_sessions
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical sessions: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return []

    def get_historical_data(self, session_id: str) -> pd.DataFrame:
        self.logger.info(
            f"🔄 Fetching historical data for session {session_id[:8]}..."
        )
        return self._paginated_fetch(session_id, "supabase_historical")

    def disconnect(self):
        """Disconnect from services."""
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
                self.connection_thread.join(timeout=5)

            self.logger.info("🔚 Disconnected from services")
        except Exception as e:
            self.logger.error(f"❌ Disconnect error: {e}")
        finally:
            self.realtime_subscriber = None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self.stats.copy()


# -----------------------------------------------------------------------------
# Data Utilities
# -----------------------------------------------------------------------------
def merge_telemetry_data(
    realtime_data: List[Dict],
    supabase_data: pd.DataFrame,
    streamlit_history: pd.DataFrame,
) -> pd.DataFrame:
    """Merge realtime + Supabase current + in-memory history."""
    try:
        all_data: List[Dict[str, Any]] = []
        if realtime_data:
            all_data.extend(realtime_data)
        if not supabase_data.empty:
            all_data.extend(supabase_data.to_dict("records"))
        if not streamlit_history.empty:
            all_data.extend(streamlit_history.to_dict("records"))

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce", utc=True
            )
            df.dropna(subset=["timestamp"], inplace=True)
        else:
            return df

        dedup_cols = ["timestamp"]
        if "message_id" in df.columns:
            dedup_cols.append("message_id")
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error merging telemetry data: {e}")
        return pd.DataFrame()


def initialize_session_state():
    defaults = {
        "telemetry_manager": None,
        "telemetry_data": pd.DataFrame(),
        "last_update": datetime.now(),
        "auto_refresh": True,
        "dynamic_charts": [],
        "data_source_mode": "realtime_session",
        "selected_session": None,
        "historical_sessions": [],
        "current_session_id": None,
        "is_viewing_historical": False,
        "pagination_info": {
            "is_loading": False,
            "current_session": None,
            "total_requests": 0,
            "total_rows": 0,
        },
        "chart_info_initialized": False,
        "data_quality_notifications": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def calculate_roll_and_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Roll and Pitch from accelerometer axes:
      roll = arctan2(ay, sqrt(ax^2 + az^2))
      pitch = arctan2(ax, sqrt(ay^2 + az^2))
    """
    if df.empty:
        return df

    df_calc = df.copy()
    accel_cols = ["accel_x", "accel_y", "accel_z"]
    if not all(col in df_calc.columns for col in accel_cols):
        return df_calc

    try:
        for col in accel_cols:
            df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce")

        denom_roll = np.sqrt(
            df_calc["accel_x"] ** 2 + df_calc["accel_z"] ** 2
        )
        denom_roll = np.where(denom_roll == 0, 1e-10, denom_roll)
        df_calc["roll_rad"] = np.arctan2(df_calc["accel_y"], denom_roll)
        df_calc["roll_deg"] = np.degrees(df_calc["roll_rad"])

        denom_pitch = np.sqrt(
            df_calc["accel_y"] ** 2 + df_calc["accel_z"] ** 2
        )
        denom_pitch = np.where(denom_pitch == 0, 1e-10, denom_pitch)
        df_calc["pitch_rad"] = np.arctan2(df_calc["accel_x"], denom_pitch)
        df_calc["pitch_deg"] = np.degrees(df_calc["pitch_rad"])

        df_calc[["roll_rad", "roll_deg", "pitch_rad", "pitch_deg"]] = (
            df_calc[["roll_rad", "roll_deg", "pitch_rad", "pitch_deg"]]
            .replace([np.inf, -np.inf, np.nan], 0.0)
        )
    except Exception as e:
        st.warning(f"⚠️ Error calculating Roll and Pitch: {e}")
        df_calc["roll_rad"] = 0.0
        df_calc["roll_deg"] = 0.0
        df_calc["pitch_rad"] = 0.0
        df_calc["pitch_deg"] = 0.0

    return df_calc


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    default_kpis = {
        "current_speed_ms": 0.0,
        "total_distance_km": 0.0,
        "max_speed_ms": 0.0,
        "avg_speed_ms": 0.0,
        "current_speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "avg_speed_kmh": 0.0,
        "total_energy_kwh": 0.0,
        "avg_power_w": 0.0,
        "efficiency_km_per_kwh": 0.0,
        "battery_voltage_v": 0.0,
        "battery_percentage": 0.0,
        "avg_current_a": 0.0,
        "current_roll_deg": 0.0,
        "current_pitch_deg": 0.0,
        "max_roll_deg": 0.0,
        "max_pitch_deg": 0.0,
    }
    if df.empty:
        return default_kpis

    try:
        df = calculate_roll_and_pitch(df)

        numeric_cols = [
            "energy_j",
            "speed_ms",
            "distance_m",
            "power_w",
            "voltage_v",
            "current_a",
            "latitude",
            "longitude",
            "altitude",
            "roll_deg",
            "pitch_deg",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        kpis = default_kpis.copy()

        if "speed_ms" in df.columns:
            s = df["speed_ms"].dropna()
            if not s.empty:
                kpis["current_speed_ms"] = max(0, s.iloc[-1])
                kpis["max_speed_ms"] = max(0, s.max())
                kpis["avg_speed_ms"] = max(0, s.mean())

        kpis["current_speed_kmh"] = kpis["current_speed_ms"] * 3.6
        kpis["max_speed_kmh"] = kpis["max_speed_ms"] * 3.6
        kpis["avg_speed_kmh"] = kpis["avg_speed_ms"] * 3.6

        if "distance_m" in df.columns and not df["distance_m"].dropna().empty:
            kpis["total_distance_km"] = max(
                0, df["distance_m"].dropna().iloc[-1] / 1000
            )

        if "energy_j" in df.columns and not df["energy_j"].dropna().empty:
            kpis["total_energy_kwh"] = max(
                0, df["energy_j"].dropna().iloc[-1] / 3_600_000
            )

        if "power_w" in df.columns:
            p = df["power_w"].dropna()
            if not p.empty:
                kpis["avg_power_w"] = max(0, p.mean())

        if kpis["total_energy_kwh"] > 0:
            kpis["efficiency_km_per_kwh"] = (
                kpis["total_distance_km"] / kpis["total_energy_kwh"]
            )

        if "voltage_v" in df.columns:
            v = df["voltage_v"].dropna()
            if not v.empty:
                kpis["battery_voltage_v"] = max(0, v.iloc[-1])
                nominal_max = 55.0
                nominal_min = 40.0
                cur = kpis["battery_voltage_v"]
                if cur > nominal_min:
                    kpis["battery_percentage"] = min(
                        100,
                        max(
                            0,
                            ((cur - nominal_min) / (nominal_max - nominal_min))
                            * 100,
                        ),
                    )

        if "current_a" in df.columns:
            c = df["current_a"].dropna()
            if not c.empty:
                kpis["avg_current_a"] = max(0.0, c.mean())

        if "roll_deg" in df.columns:
            r = df["roll_deg"].dropna()
            if not r.empty:
                kpis["current_roll_deg"] = r.iloc[-1]
                kpis["max_roll_deg"] = r.abs().max()

        if "pitch_deg" in df.columns:
            p = df["pitch_deg"].dropna()
            if not p.empty:
                kpis["current_pitch_deg"] = p.iloc[-1]
                kpis["max_pitch_deg"] = p.abs().max()

        return kpis
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def create_small_gauge(
    value: float,
    max_val: Optional[float],
    title: str,
    color: str,
    suffix: str = "",
    avg_ref: Optional[float] = None,
    thresh_val: Optional[float] = None,
) -> go.Figure:
    """
    Plotly Indicator gauge: dial+number (+optional delta & threshold).
    See Plotly indicator docs for supported options.
    """
    if max_val is None or max_val <= 0:
        max_val = value * 1.2 if value > 0 else 1.0

    rgb = mcolors.to_rgb(color)
    steps = [
        {"range": [0, max_val * 0.6], "color": f"rgba{(*rgb, 0.12)}"},
        {"range": [max_val * 0.6, max_val], "color": f"rgba{(*rgb, 0.28)}"},
    ]

    gauge = {
        "axis": {
            "range": [0, max_val],
            "ticks": "inside",
            "showticklabels": False,
            "tickcolor": "rgba(128,128,128,0.3)",
        },
        "bar": {"color": color, "thickness": 0.4},
        "steps": steps,
        "bgcolor": "rgba(0,0,0,0)",
    }

    if thresh_val is not None:
        gauge["threshold"] = {
            "line": {"color": "rgba(220,53,69,0.85)", "width": 3},
            "thickness": 0.85,
            "value": thresh_val,
        }

    mode = "gauge+number" + ("+delta" if avg_ref is not None else "")
    delta_cfg = (
        {
            "reference": avg_ref,
            "position": "top",
            "increasing": {"color": color},
            "font": {"size": 12},
        }
        if avg_ref is not None
        else None
    )

    fig = go.Figure(
        go.Indicator(
            mode=mode,
            value=value,
            number={"suffix": suffix, "font": {"size": 18}},
            delta=delta_cfg,
            gauge=gauge,
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=140,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_live_gauges(kpis: Dict[str, float], unique_ns: str = "gauges"):
    st.markdown("##### 📊 Live Performance Gauges")

    st.markdown('<div class="widget-grid">', unsafe_allow_html=True)
    cols = st.columns(6)

    with cols[0]:
        st.markdown(
            '<div class="gauge-container"><div class="gauge-title">'
            "🚀 Speed (km/h)</div>",
            unsafe_allow_html=True,
        )
        speed_fig = create_small_gauge(
            kpis["current_speed_kmh"],
            max(100, kpis["max_speed_kmh"] + 5),
            "Speed",
            "#1f77b4",
            " km/h",
        )
        st.plotly_chart(
            speed_fig, use_container_width=True, key=f"{unique_ns}_speed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown(
            '<div class="gauge-container"><div class="gauge-title">'
            "🔋 Battery (%)</div>",
            unsafe_allow_html=True,
        )
        battery_fig = create_small_gauge(
            kpis["battery_percentage"], 100, "Battery", "#2ca02c", "%"
        )
        st.plotly_chart(
            battery_fig, use_container_width=True, key=f"{unique_ns}_battery"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[2]:
        st.markdown(
            '<div class="gauge-container"><div class="gauge-title">'
            "💡 Power (W)</div>",
            unsafe_allow_html=True,
        )
        power_fig = create_small_gauge(
            kpis["avg_power_w"],
            max(1000, kpis["avg_power_w"] * 2),
            "Power",
            "#ff7f0e",
            " W",
        )
        st.plotly_chart(
            power_fig, use_container_width=True, key=f"{unique_ns}_power"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[3]:
        st.markdown(
            '<div class="gauge-container"><div class="gauge-title">'
            "♻️ Efficiency (km/kWh)</div>",
            unsafe_allow_html=True,
        )
        eff_val = kpis["efficiency_km_per_kwh"]
        eff_fig = create_small_gauge(
            eff_val, max(100, eff_val * 1.5) if eff_val > 0 else 100, "Eff",
            "#6a51a3", ""
        )
        st.plotly_chart(
            eff_fig, use_container_width=True, key=f"{unique_ns}_eff"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[4]:
        st.markdown(
            '<div class="gauge-container"><div class="gauge-title">'
            "🔄 Roll (°)</div>",
            unsafe_allow_html=True,
        )
        roll_max = max(
            45, abs(kpis["current_roll_deg"]) + 10
        ) if kpis["current_roll_deg"] != 0 else 45
        roll_fig = create_small_gauge(
            kpis["current_roll_deg"], roll_max, "Roll", "#e377c2", "°"
        )
        st.plotly_chart(
            roll_fig, use_container_width=True, key=f"{unique_ns}_roll"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[5]:
        st.markdown(
            '<div class="gauge-container"><div class="gauge-title">'
            "📐 Pitch (°)</div>",
            unsafe_allow_html=True,
        )
        pitch_max = max(
            45, abs(kpis["current_pitch_deg"]) + 10
        ) if kpis["current_pitch_deg"] != 0 else 45
        pitch_fig = create_small_gauge(
            kpis["current_pitch_deg"], pitch_max, "Pitch", "#17becf", "°"
        )
        st.plotly_chart(
            pitch_fig, use_container_width=True, key=f"{unique_ns}_pitch"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_kpi_header(
    kpis: Dict[str, float], unique_ns: str = "kpiheader", show_gauges: bool = True
):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📏 Distance", f"{kpis['total_distance_km']:.2f} km", border=True
        )
        st.metric(
            "🏃 Max Speed", f"{kpis['max_speed_kmh']:.1f} km/h", border=True
        )

    with col2:
        st.metric(
            "⚡ Avg Speed", f"{kpis['avg_speed_kmh']:.1f} km/h", border=True
        )
        st.metric("🔋 Energy", f"{kpis['total_energy_kwh']:.2f} kWh", border=True)

    with col3:
        st.metric(
            "⚡ Voltage", f"{kpis['battery_voltage_v']:.1f} V", border=True
        )
        st.metric(
            "🔄 Current Roll", f"{kpis['current_roll_deg']:.1f}°", border=True
        )

    with col4:
        st.metric(
            "💡 Avg Power", f"{kpis['avg_power_w']:.1f} W", border=True
        )
        st.metric(
            "📐 Current Pitch", f"{kpis['current_pitch_deg']:.1f}°", border=True
        )

    if show_gauges:
        render_live_gauges(kpis, unique_ns)


def render_overview_tab(kpis: Dict[str, float]):
    st.markdown("### 📊 Performance Overview")
    st.markdown(
        "Real-time key performance indicators for your Shell Eco-marathon vehicle"
    )
    render_kpi_header(kpis, unique_ns="overview", show_gauges=True)


def render_session_info(session_data: Dict[str, Any]):
    st.markdown(
        f"""
    <div class="card" style="padding: 1rem;">
        <h3 style="margin: 0 0 .5rem 0; color: var(--accent);">
            📊 Session Information
        </h3>
        <p>📋 <strong>Session:</strong> {session_data['session_id'][:8]}...</p>
        <p>📅 <strong>Start:</strong>
           {session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>⏱️ <strong>Duration:</strong>
           {str(session_data['duration']).split('.')[0]}</p>
        <p>📊 <strong>Records:</strong>
           {session_data['record_count']:,}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def analyze_data_quality(df: pd.DataFrame, is_realtime: bool):
    """
    Basic anomaly checks: stale stream and static sensors.
    """
    if df.empty or len(df) < 10:
        st.session_state.data_quality_notifications = []
        return

    notifications: List[str] = []
    logger = logging.getLogger("TelemetryDashboard")

    # Stale data detection
    if is_realtime:
        try:
            last_ts = df["timestamp"].iloc[-1]
            now_utc = datetime.now(timezone.utc)
            dt = (now_utc - last_ts).total_seconds()
            if len(df) > 2:
                diffs = df["timestamp"].diff().dt.total_seconds().dropna()
                avg_rate = diffs.tail(20).mean()
                if pd.isna(avg_rate) or avg_rate <= 0:
                    avg_rate = 1.0
            else:
                avg_rate = 1.0
            threshold = max(5.0, avg_rate * 5)
            if dt > threshold:
                notifications.append(
                    "🚨 **Data Stream Stalled:** No new data for "
                    f"{int(dt)}s. (Expected every ~{avg_rate:.1f}s)"
                )
        except Exception as e:
            logger.warning(f"Stale data check failed: {e}")

    recent_df = df.tail(15)
    sensors_to_check = [
        "latitude",
        "longitude",
        "altitude",
        "voltage_v",
        "current_a",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_x",
        "accel_y",
        "accel_z",
    ]

    failing_sensors = []
    all_fail = True
    for col in sensors_to_check:
        if col in recent_df.columns:
            s = recent_df[col].dropna()
            fail = False
            if len(s) < 5:
                all_fail = False
                continue
            if s.abs().max() < 1e-6 or s.std() < 1e-6:
                fail = True
            if fail:
                failing_sensors.append(col)
            else:
                all_fail = False
        else:
            all_fail = False

    if all_fail and len(failing_sensors) > 3:
        notifications.append(
            "🚨 **Critical Alert:** Multiple sensors (e.g., "
            f"{', '.join(failing_sensors[:3])}...) are static or zero."
        )
    elif failing_sensors:
        notifications.append(
            "⚠️ **Sensor Anomaly:** Static/zero values detected: "
            f"**{', '.join(failing_sensors)}**."
        )

    st.session_state.data_quality_notifications = notifications


def create_speed_chart(df: pd.DataFrame) -> go.Figure:
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
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_power_chart(df: pd.DataFrame) -> go.Figure:
    needed = {"voltage_v", "current_a", "power_w"}
    if df.empty or not needed.issubset(df.columns):
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
    fig.update_layout(
        height=500,
        title_text="⚡ Electrical System Performance",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_imu_chart(df: pd.DataFrame) -> go.Figure:
    needed = {
        "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"
    }
    if df.empty or not needed.issubset(df.columns):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    df = calculate_roll_and_pitch(df)
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "🎯 Gyroscope Data (deg/s)",
            "📈 Accelerometer Data (m/s²)",
            "🎭 Roll & Pitch (degrees)",
        ),
        vertical_spacing=0.15,
    )
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
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["roll_deg"],
            name="Roll (°)",
            line=dict(color="#e377c2", width=3),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["pitch_deg"],
            name="Pitch (°)",
            line=dict(color="#17becf", width=3),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        height=700,
        title_text="⚡ IMU System Performance with Roll & Pitch",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_imu_detail_chart(df: pd.DataFrame) -> go.Figure:
    needed = {
        "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"
    }
    if df.empty or not needed.issubset(df.columns):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    df = calculate_roll_and_pitch(df)
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "🌀 Gyro X",
            "🌀 Gyro Y",
            "🌀 Gyro Z",
            "📊 Accel X",
            "📊 Accel Y",
            "📊 Accel Z",
            "🔄 Roll (°)",
            "📐 Pitch (°)",
            "🎯 R&P Combined",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]

    for i, (axis, color) in enumerate(
        zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)
    ):
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

    # Roll
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["roll_deg"],
            name="Roll",
            line=dict(color="#e377c2", width=3),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Pitch
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["pitch_deg"],
            name="Pitch",
            line=dict(color="#17becf", width=3),
            showlegend=False,
        ),
        row=3,
        col=2,
    )

    # Combined Roll & Pitch
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["roll_deg"],
            name="Roll",
            line=dict(color="#e377c2", width=2),
            showlegend=False,
        ),
        row=3,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["pitch_deg"],
            name="Pitch",
            line=dict(color="#17becf", width=2),
            showlegend=False,
        ),
        row=3,
        col=3,
    )

    fig.update_layout(
        height=700,
        title_text="🎮 Detailed IMU Sensor Analysis with Roll & Pitch",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_efficiency_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or not {"speed_ms", "power_w"}.issubset(df.columns):
        return go.Figure().add_annotation(
            text="No efficiency data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
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
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_gps_map_with_altitude(df: pd.DataFrame) -> go.Figure:
    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    df_valid = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3],
        subplot_titles=("🛰️ Vehicle Track", "⛰️ Altitude Profile"),
        specs=[[{"type": "scattermapbox"}, {"type": "scatter"}]],
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=df_valid["latitude"],
            lon=df_valid["longitude"],
            mode="markers+lines",
            marker=dict(
                size=8,
                color=df_valid["speed_ms"] if "speed_ms" in df_valid.columns else "#1f77b4",
                colorscale="plasma",
                showscale=True,
                colorbar=dict(title="Speed (m/s)", x=0.65),
            ),
            line=dict(width=2, color="#1f77b4"),
            name="Track",
        ),
        row=1, col=1
    )

    if "altitude" in df.columns:
        alt_data = df_valid[df_valid["altitude"] != 0]
        if not alt_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=alt_data["timestamp"],
                    y=alt_data["altitude"],
                    mode="lines",
                    line=dict(color="#2ca02c", width=2),
                    name="Altitude",
                ),
                row=1, col=2
            )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(
            lat=df_valid["latitude"].mean(),
            lon=df_valid["longitude"].mean()
        ), zoom=14),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------
# Dynamic Charts Section
# ---------------------------
def get_available_columns(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]


def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]) -> go.Figure:
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    try:
        if chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == "bar":
            fig = px.bar(df.tail(20), x=x_col, y=y_col, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=y_col, title=f"Distribution of {y_col}")
        elif chart_type == "heatmap":
            numeric_cols = get_available_columns(df)
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="🔥 Correlation Heatmap",
                                color_continuous_scale="RdBu_r")
            else:
                return go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title)
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_dynamic_charts_section(df: pd.DataFrame):
    available_columns = get_available_columns(df)
    if not available_columns:
        st.warning("⏳ No numeric data available for creating charts.")
        return

    if st.button("➕ Add Chart", key="add_chart_btn"):
        new_chart = {
            "id": str(uuid.uuid4()),
            "title": "New Chart",
            "chart_type": "line",
            "x_axis": "timestamp" if "timestamp" in df.columns else available_columns[0],
            "y_axis": available_columns[0],
        }
        st.session_state.dynamic_charts.append(new_chart)
        st.rerun()

    for i, chart_config in enumerate(list(st.session_state.dynamic_charts)):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])
            with col1:
                st.session_state.dynamic_charts[i]["title"] = st.text_input(
                    "Title", value=chart_config.get("title", "New Chart"),
                    key=f"title_{chart_config['id']}"
                )
            with col2:
                st.session_state.dynamic_charts[i]["chart_type"] = st.selectbox(
                    "Type", ["line", "scatter", "bar", "histogram", "heatmap"],
                    index=["line", "scatter", "bar", "histogram", "heatmap"].index(
                        chart_config.get("chart_type", "line")
                    ),
                    key=f"type_{chart_config['id']}"
                )
            with col3:
                if chart_config["chart_type"] not in ["histogram", "heatmap"]:
                    st.session_state.dynamic_charts[i]["x_axis"] = st.selectbox(
                        "X-Axis", ["timestamp"] + available_columns,
                        index=(["timestamp"] + available_columns).index(
                            chart_config.get("x_axis", "timestamp")
                        ),
                        key=f"x_{chart_config['id']}"
                    )
            with col4:
                if chart_config["chart_type"] != "heatmap":
                    st.session_state.dynamic_charts[i]["y_axis"] = st.selectbox(
                        "Y-Axis", available_columns,
                        index=available_columns.index(
                            chart_config.get("y_axis", available_columns[0])
                        ),
                        key=f"y_{chart_config['id']}"
                    )
            with col5:
                if st.button("🗑️", key=f"delete_{chart_config['id']}"):
                    st.session_state.dynamic_charts.pop(i)
                    st.rerun()

            fig = create_dynamic_chart(df, chart_config)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Main App
# ---------------------------
def main():
    st.markdown('<div class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</div>', unsafe_allow_html=True)
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("🔧 Connection & Data Source")
        data_source_mode = st.radio(
            "📊 Data Source",
            options=["realtime_session", "historical"],
            format_func=lambda x: "🔴 Real-time + Session Data" if x == "realtime_session" else "📚 Historical Data",
            key="data_source_mode_radio",
        )
        if data_source_mode != st.session_state.data_source_mode:
            st.session_state.data_source_mode = data_source_mode
            st.session_state.telemetry_data = pd.DataFrame()
            st.session_state.is_viewing_historical = (data_source_mode == "historical")
            st.session_state.selected_session = None
            st.session_state.current_session_id = None
            st.rerun()

        if st.session_state.data_source_mode == "realtime_session":
            if st.button("🔌 Connect"):
                st.session_state.telemetry_manager = EnhancedTelemetryManager()
                st.session_state.telemetry_manager.connect_supabase()
                if ABLY_AVAILABLE:
                    st.session_state.telemetry_manager.connect_realtime()
                st.rerun()
            if st.button("🛑 Disconnect"):
                if st.session_state.telemetry_manager:
                    st.session_state.telemetry_manager.disconnect()
                st.rerun()
        else:
            if not st.session_state.telemetry_manager:
                st.session_state.telemetry_manager = EnhancedTelemetryManager()
                st.session_state.telemetry_manager.connect_supabase()
            if st.button("🔄 Refresh Sessions"):
                st.session_state.historical_sessions = st.session_state.telemetry_manager.get_historical_sessions()
                st.rerun()
            if st.session_state.historical_sessions:
                selected_idx = st.selectbox(
                    "📋 Select Session",
                    options=range(len(st.session_state.historical_sessions)),
                    format_func=lambda x: f"{st.session_state.historical_sessions[x]['session_id'][:8]}... - {st.session_state.historical_sessions[x]['start_time'].strftime('%Y-%m-%d %H:%M')}",
                )
                selected_session = st.session_state.historical_sessions[selected_idx]
                if st.session_state.selected_session != selected_session:
                    st.session_state.selected_session = selected_session
                    st.session_state.telemetry_data = st.session_state.telemetry_manager.get_historical_data(selected_session["session_id"])
                    st.rerun()

    # Main content
    df = st.session_state.telemetry_data.copy()
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        return

    analyze_data_quality(df, is_realtime=(st.session_state.data_source_mode == "realtime_session"))
    for msg in st.session_state.data_quality_notifications:
        if "🚨" in msg:
            st.error(msg)
        else:
            st.warning(msg)

    kpis = calculate_kpis(df)
    tabs = st.tabs(["📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", "🎮 IMU Detail", "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data"])

    with tabs[0]:
        render_overview_tab(kpis)
    with tabs[1]:
        render_live_gauges(kpis, "speedtab")
        st.plotly_chart(create_speed_chart(df), use_container_width=True)
    with tabs[2]:
        render_live_gauges(kpis, "powertab")
        st.plotly_chart(create_power_chart(df), use_container_width=True)
    with tabs[3]:
        render_live_gauges(kpis, "imutab")
        st.plotly_chart(create_imu_chart(df), use_container_width=True)
    with tabs[4]:
        render_live_gauges(kpis, "imudetailtab")
        st.plotly_chart(create_imu_detail_chart(df), use_container_width=True)
    with tabs[5]:
        render_live_gauges(kpis, "efftab")
        st.plotly_chart(create_efficiency_chart(df), use_container_width=True)
    with tabs[6]:
        render_live_gauges(kpis, "gpstab")
        st.plotly_chart(create_gps_map_with_altitude(df), use_container_width=True)
    with tabs[7]:
        render_dynamic_charts_section(df)
    with tabs[8]:
        st.dataframe(df.tail(100), use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "telemetry.csv", "text/csv")


if __name__ == "__main__":
    main()
