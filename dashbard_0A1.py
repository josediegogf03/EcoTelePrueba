# telem_dashboard_refreshed.py
# Visual refresh with gauges and unique keys, preserving all functionality.

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
import math

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

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

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Configuration (unchanged)
DASHBOARD_ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
DASHBOARD_CHANNEL_NAME = "telemetry-dashboard-channel"
SUPABASE_URL = "https://dsfmdziehhgmrconjcns.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRzZm1kemllaGhnbXJjb25qY25zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5MDEyOTIsImV4cCI6MjA2NzQ3NzI5Mn0.P41bpLkP0tKpTktLx6hFOnnyrAB9N_yihQP1v6zTRwc"
SUPABASE_TABLE_NAME = "telemetry"
SUPABASE_MAX_ROWS_PER_REQUEST = 1000
MAX_DATAPOINTS_PER_SESSION = 1000000

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

# THEME-AWARE STYLING: subtle blurred gradient bottom background, rounded soft cards,
# dark/light optimized; no full solid background; gradient anchored bottom.
st.markdown(
    """
<style>
:root {
  --primary: #1f77b4;
  --accent: #ff7f0e;
  --success: #2ca02c;
  --warn: #ffc107;
  --error: #d62728;

  --bg: light-dark(#ffffff, #0e1117);
  --bg-card: light-dark(#f8f9fa, #161a23);
  --bg-soft: light-dark(#ffffff, #0e1117);
  --text: light-dark(#1d232a, #eaeef4);
  --muted: light-dark(#6c757d, #a0a0a0);
  --border: light-dark(#e2e8f0, #2b3340);
  --shadow: light-dark(0 10px 25px rgba(0,0,0,.08), 0 10px 25px rgba(0,0,0,.35));
  --ring: light-dark( rgba(31,119,180,.15), rgba(31,119,180,.25) );
  --radius: 16px;
}

html, body, .block-container {
  background: transparent !important;
  color: var(--text);
}

/* Bottom anchored, semi-transparent radial gradient, does not reach top */
.main:before {
  content: "";
  position: fixed;
  inset: auto 0 0 0;
  height: 55vh;
  pointer-events: none;
  background:
    radial-gradient(60% 60% at 50% 100%,
      rgba(31, 119, 180, 0.22) 0%,
      rgba(31, 119, 180, 0.12) 30%,
      rgba(255, 127, 14, 0.10) 60%,
      rgba(0,0,0,0) 90%);
  filter: blur(40px);
  z-index: -1;
}

/* Cards / containers */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1rem 1.25rem;
}
.card.soft {
  background: linear-gradient(180deg, var(--bg-card), var(--bg-soft));
}

/* Header */
.main-header {
  font-size: 2.0rem;
  color: var(--primary);
  text-align: center;
  font-weight: 800;
  margin: 0 0 1rem 0;
  text-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

/* Status badges */
.status-indicator {
  display: flex; align-items: center; justify-content: center;
  border-radius: 999px; padding: 0.5rem 0.9rem;
  gap: 0.5rem; font-weight: 700; border: 1px solid var(--border);
  background: var(--bg-card);
  box-shadow: var(--shadow);
}
.status-connected { color: #0f9d58; border-color: rgba(15,157,88,0.35); }
.status-disconnected { color: #ea4335; border-color: rgba(234,67,53,0.35); }
.status-historical { color: #5f6368; }

/* KPI grid wrap */
.kpi-wrap {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 0.75rem;
}

/* Tiny gauge holder */
.gauge-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.5rem 0.75rem;
  box-shadow: var(--shadow);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--border);
}

/* Buttons */
.stButton > button {
  border-radius: 10px;
  border: 1px solid var(--primary);
  background: var(--primary);
  color: #fff;
  font-weight: 700;
  box-shadow: var(--shadow);
}
.stButton > button:hover {
  background: transparent;
  color: var(--primary);
}

/* Info bars */
.notice {
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  padding: 0.8rem 1rem;
  color: var(--text);
  background: var(--bg-card);
}

/* Make tables and metrics align better */
[data-testid="stMetric"] {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--bg-card);
  padding: 0.75rem;
  box-shadow: var(--shadow);
}

</style>
""",
    unsafe_allow_html=True,
)

# Logger
def setup_terminal_logging():
    logger = logging.getLogger("TelemetryDashboard")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

setup_terminal_logging()

# ----- Core classes and functions from your original code (functionality unchanged) -----

class EnhancedTelemetryManager:
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
                "sessions_paginated": 0,
            },
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self.logger = logging.getLogger("TelemetryDashboard")

    def connect_supabase(self) -> bool:
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
        messages = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
        return messages

    def _paginated_fetch(self, session_id: str, data_source: str = "supabase_current") -> pd.DataFrame:
        try:
            if not self.supabase_client:
                self.logger.error("❌ Supabase client not initialized")
                return pd.DataFrame()

            all_data = []
            offset = 0
            total_fetched = 0
            request_count = 0

            self.logger.info(f"🔄 Starting paginated fetch for session {session_id[:8]}...")

            while offset < MAX_DATAPOINTS_PER_SESSION:
                try:
                    range_end = offset + SUPABASE_MAX_ROWS_PER_REQUEST - 1
                    self.logger.info(f"📄 Fetching page {request_count + 1}: rows {offset}-{range_end}")
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
                        self.logger.info(f"✅ No more data found at offset {offset}")
                        break
                    batch_size = len(response.data)
                    all_data.extend(response.data)
                    total_fetched += batch_size
                    self.logger.info(f"📊 Fetched {batch_size} rows (total: {total_fetched})")
                    if batch_size < SUPABASE_MAX_ROWS_PER_REQUEST:
                        self.logger.info("✅ Reached end of data")
                        break
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"❌ Error in pagination request {request_count}: {e}")
                    offset += SUPABASE_MAX_ROWS_PER_REQUEST
                    continue

            with self._lock:
                ps = self.stats["pagination_stats"]
                ps["total_requests"] += request_count
                ps["total_rows_fetched"] += total_fetched
                ps["largest_session_size"] = max(ps["largest_session_size"], total_fetched)
                if request_count > 1:
                    ps["sessions_paginated"] += 1

            if all_data:
                df = pd.DataFrame(all_data)
                df["data_source"] = data_source
                self.logger.info(
                    f"✅ Successfully fetched {len(df)} total rows for session {session_id[:8]}..."
                )
                return df
            else:
                self.logger.warning(f"⚠️ No data found for session {session_id}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ Error in paginated fetch for session {session_id}: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return pd.DataFrame()

    def get_current_session_data(self, session_id: str) -> pd.DataFrame:
        self.logger.info(f"🔄 Fetching current session data for {session_id[:8]}...")
        return self._paginated_fetch(session_id, "supabase_current")

    def get_historical_sessions(self) -> List[Dict[str, Any]]:
        try:
            if not self.supabase_client:
                self.logger.error("❌ Supabase client not initialized")
                return []

            self.logger.info("🔄 Fetching historical sessions list...")
            all_records = []
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
                    self.logger.error(f"❌ Error fetching session records at offset {offset}: {e}")
                    break

            if not all_records:
                self.logger.warning("⚠️ No session records found")
                return []

            sessions = {}
            for record in all_records:
                sid = record["session_id"]
                ts = record["timestamp"]
                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid,
                        "start_time": ts,
                        "end_time": ts,
                        "record_count": 1,
                    }
                else:
                    sessions[sid]["record_count"] += 1
                    if ts < sessions[sid]["start_time"]:
                        sessions[sid]["start_time"] = ts
                    if ts > sessions[sid]["end_time"]:
                        sessions[sid]["end_time"] = ts

            session_list = []
            for s in sessions.values():
                try:
                    start_dt = datetime.fromisoformat(s["start_time"].replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(s["end_time"].replace("Z", "+00:00"))
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
                    self.logger.error(f"❌ Error processing session {s['session_id']}: {e}")

            sorted_sessions = sorted(session_list, key=lambda x: x["start_time"], reverse=True)
            self.logger.info(f"✅ Found {len(sorted_sessions)} unique sessions")
            return sorted_sessions

        except Exception as e:
            self.logger.error(f"❌ Error fetching historical sessions: {e}")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
            return []

    def get_historical_data(self, session_id: str) -> pd.DataFrame:
        self.logger.info(f"🔄 Fetching historical data for session {session_id[:8]}...")
        return self._paginated_fetch(session_id, "supabase_historical")

    def disconnect(self):
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

def merge_telemetry_data(
    realtime_data: List[Dict],
    supabase_data: pd.DataFrame,
    streamlit_history: pd.DataFrame,
) -> pd.DataFrame:
    try:
        all_data = []
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df.dropna(subset=["timestamp"], inplace=True)
        else:
            return df
        dedup_columns = ["timestamp"]
        if "message_id" in df.columns:
            dedup_columns.append("message_id")
        df = df.drop_duplicates(subset=dedup_columns, keep="last")
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
    }
    if df.empty:
        return default_kpis

    try:
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
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        kpis = default_kpis.copy()

        if "speed_ms" in df.columns:
            speed_data = df["speed_ms"].dropna()
            if not speed_data.empty:
                kpis["current_speed_ms"] = max(0, speed_data.iloc[-1])
                kpis["max_speed_ms"] = max(0, speed_data.max())
                kpis["avg_speed_ms"] = max(0, speed_data.mean())

        kpis["current_speed_kmh"] = kpis["current_speed_ms"] * 3.6
        kpis["max_speed_kmh"] = kpis["max_speed_ms"] * 3.6
        kpis["avg_speed_kmh"] = kpis["avg_speed_ms"] * 3.6

        if "distance_m" in df.columns and not df["distance_m"].dropna().empty:
            kpis["total_distance_km"] = max(0, df["distance_m"].dropna().iloc[-1] / 1000)

        if "energy_j" in df.columns and not df["energy_j"].dropna().empty:
            kpis["total_energy_kwh"] = max(0, df["energy_j"].dropna().iloc[-1] / 3_600_000)

        if "power_w" in df.columns:
            power_data = df["power_w"].dropna()
            if not power_data.empty:
                kpis["avg_power_w"] = max(0, power_data.mean())

        if kpis["total_energy_kwh"] > 0:
            kpis["efficiency_km_per_kwh"] = (
                kpis["total_distance_km"] / kpis["total_energy_kwh"]
            )

        if "voltage_v" in df.columns:
            voltage_data = df["voltage_v"].dropna()
            if not voltage_data.empty:
                kpis["battery_voltage_v"] = max(0, voltage_data.iloc[-1])
                nominal_voltage = 48.0
                max_voltage = 55.0
                min_voltage = 40.0
                current_voltage = kpis["battery_voltage_v"]
                if current_voltage > min_voltage:
                    kpis["battery_percentage"] = min(
                        100,
                        max(
                            0,
                            ((current_voltage - min_voltage) / (max_voltage - min_voltage))
                            * 100,
                        ),
                    )

        return kpis
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return default_kpis

def render_kpi_header(kpis: Dict[str, float], key_prefix: str = "kpihdr"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚀 Current Speed", f"{kpis['current_speed_kmh']:.1f} km/h")
        st.metric("📏 Distance", f"{kpis['total_distance_km']:.2f} km")
    with col2:
        st.metric("🏃 Max Speed", f"{kpis['max_speed_kmh']:.1f} km/h")
        st.metric("⚡ Avg Speed", f"{kpis['avg_speed_kmh']:.1f} km/h")
    with col3:
        st.metric("🔋 Energy", f"{kpis['total_energy_kwh']:.2f} kWh")
        st.metric("💡 Avg Power", f"{kpis['avg_power_w']:.1f} W")
    with col4:
        battery_display = f"{kpis['battery_voltage_v']:.1f}V ({kpis['battery_percentage']:.0f}%)"
        st.metric("🔋 Battery", battery_display)
        st.metric("♻️ Efficiency", f"{kpis['efficiency_km_per_kwh']:.2f} km/kWh")

# NEW: Compact Plotly Gauges (small graphs)
def render_small_gauges(kpis: Dict[str, float], key_prefix: str = "gauge"):
    # Speed gauge (km/h)
    speed_val = max(0, float(kpis.get("current_speed_kmh", 0.0)))
    speed_max = max(50, float(kpis.get("max_speed_kmh", 0.0)) + 5)

    fig_speed = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=speed_val,
            number={"suffix": " km/h", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, speed_max]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, speed_max * 0.5], "color": "rgba(31,119,180,0.12)"},
                    {"range": [speed_max * 0.5, speed_max * 0.8], "color": "rgba(255,127,14,0.12)"},
                    {"range": [speed_max * 0.8, speed_max], "color": "rgba(214,39,40,0.12)"},
                ],
                "threshold": {"line": {"color": "#ff7f0e", "width": 3}, "value": speed_val},
            },
            title={"text": "Speed", "font": {"size": 14}},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig_speed.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")

    # Battery %
    batt_pct = float(kpis.get("battery_percentage", 0.0))
    fig_batt = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=batt_pct,
            number={"suffix": " %", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ca02c"},
                "steps": [
                    {"range": [0, 20], "color": "rgba(214,39,40,0.15)"},
                    {"range": [20, 60], "color": "rgba(255,127,14,0.15)"},
                    {"range": [60, 100], "color": "rgba(44,160,44,0.15)"},
                ],
                "threshold": {"line": {"color": "#2ca02c", "width": 3}, "value": batt_pct},
            },
            title={"text": "Battery %", "font": {"size": 14}},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig_batt.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")

    # Avg Power (W)
    avg_power = float(kpis.get("avg_power_w", 0.0))
    pow_max = max(100, avg_power * 2)
    fig_pow = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_power,
            number={"suffix": " W", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, pow_max]},
                "bar": {"color": "#ff7f0e"},
                "steps": [
                    {"range": [0, pow_max * 0.5], "color": "rgba(255,127,14,0.12)"},
                    {"range": [pow_max * 0.5, pow_max], "color": "rgba(255,127,14,0.2)"},
                ],
                "threshold": {"line": {"color": "#ff7f0e", "width": 3}, "value": avg_power},
            },
            title={"text": "Avg Power", "font": {"size": 14}},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig_pow.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")

    # Efficiency (km/kWh)
    eff = float(kpis.get("efficiency_km_per_kwh", 0.0))
    eff_max = max(10, eff * 2 if eff > 0 else 10)
    fig_eff = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=eff,
            number={"suffix": " km/kWh", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, eff_max]},
                "bar": {"color": "#17becf"},
                "steps": [
                    {"range": [0, eff_max * 0.4], "color": "rgba(23,190,207,0.12)"},
                    {"range": [eff_max * 0.4, eff_max], "color": "rgba(23,190,207,0.18)"},
                ],
                "threshold": {"line": {"color": "#17becf", "width": 3}, "value": eff},
            },
            title={"text": "Efficiency", "font": {"size": 14}},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig_eff.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.plotly_chart(fig_speed, use_container_width=True, key=f"{key_prefix}_speed")
    with c2:
        st.plotly_chart(fig_batt, use_container_width=True, key=f"{key_prefix}_batt")
    with c3:
        st.plotly_chart(fig_pow, use_container_width=True, key=f"{key_prefix}_pow")
    with c4:
        st.plotly_chart(fig_eff, use_container_width=True, key=f"{key_prefix}_eff")

def render_overview_tab(kpis: Dict[str, float], key_prefix: str = "ov"):
    st.markdown("### 📊 Performance Overview")
    st.markdown("Real-time key performance indicators for your Shell Eco-marathon vehicle")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="🚀 Current Speed",
            value=f"{kpis['current_speed_kmh']:.1f} km/h",
            help="Current vehicle speed",
        )
        st.metric(
            label="🛣️ Total Distance",
            value=f"{kpis['total_distance_km']:.2f} km",
            help="Distance traveled during the session",
        )
    with col2:
        st.metric(
            label="🏃 Maximum Speed",
            value=f"{kpis['max_speed_kmh']:.1f} km/h",
            help="Highest speed achieved",
        )
        st.metric(
            label="⚡ Average Speed",
            value=f"{kpis['avg_speed_kmh']:.1f} km/h",
            help="Mean speed throughout the session",
        )
    with col3:
        st.metric(
            label="🔋 Energy Consumed",
            value=f"{kpis['total_energy_kwh']:.2f} kWh",
            help="Total energy consumption",
        )
        st.metric(
            label="💡 Average Power",
            value=f"{kpis['avg_power_w']:.1f} W",
            help="Mean power consumption",
        )
    with col4:
        battery_display = f"{kpis['battery_voltage_v']:.1f}V ({kpis['battery_percentage']:.0f}%)"
        st.metric(
            label="🔋 Battery Status",
            value=battery_display,
            help="Current battery voltage and estimated percentage",
        )
        st.metric(
            label="♻️ Efficiency",
            value=f"{kpis['efficiency_km_per_kwh']:.2f} km/kWh",
            help="Energy efficiency ratio",
        )
    # small gauges row
    st.markdown("#### 📟 Live Gauges")
    with st.container():
        render_small_gauges(kpis, key_prefix=f"{key_prefix}_gauges")

def render_session_info(session_data: Dict[str, Any]):
    st.markdown(
        f"""
    <div class="card soft">
        <h4 style="margin:0 0 .5rem 0;color:var(--primary);">📊 Session</h4>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.5rem;">
            <div><b>Session:</b> {session_data['session_id'][:8]}...</div>
            <div><b>Start:</b> {session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div><b>Duration:</b> {str(session_data['duration']).split('.')[0]}</div>
            <div><b>Records:</b> {session_data['record_count']:,}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def analyze_data_quality(df: pd.DataFrame, is_realtime: bool):
    if df.empty or len(df) < 10:
        st.session_state.data_quality_notifications = []
        return
    notifications = []
    logger = logging.getLogger("TelemetryDashboard")

    if is_realtime:
        try:
            last_timestamp = df["timestamp"].iloc[-1]
            now_utc = datetime.now(timezone.utc)
            time_since_last = (now_utc - last_timestamp).total_seconds()
            if len(df) > 2:
                time_diffs = df["timestamp"].diff().dt.total_seconds().dropna()
                avg_rate = time_diffs.tail(20).mean()
                if pd.isna(avg_rate) or avg_rate <= 0:
                    avg_rate = 1.0
            else:
                avg_rate = 1.0
            threshold = max(5.0, avg_rate * 5)
            if time_since_last > threshold:
                notifications.append(
                    f"🚨 **Data Stream Stalled:** No new data received for {int(time_since_last)}s. "
                    f"(Expected ~{avg_rate:.1f}s)"
                )
        except Exception as e:
            logger.warning(f"Could not perform stale data check: {e}")

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
    all_sensors_failing = True

    for col in sensors_to_check:
        if col in recent_df.columns:
            sensor_data = recent_df[col].dropna()
            is_failing = False
            if len(sensor_data) < 5:
                all_sensors_failing = False
                continue
            if sensor_data.abs().max() < 1e-6 or sensor_data.std() < 1e-6:
                is_failing = True
            if is_failing:
                failing_sensors.append(col)
            else:
                all_sensors_failing = False
        else:
            all_sensors_failing = False

    if all_sensors_failing and len(failing_sensors) > 3:
        notifications.append(
            "🚨 **Critical Alert:** Multiple sensors (including "
            f"{', '.join(failing_sensors[:3])}...) are static/zero."
        )
    elif failing_sensors:
        notifications.append(
            f"⚠️ **Sensor Anomaly:** Static/zero values: **{', '.join(failing_sensors)}**."
        )

    st.session_state.data_quality_notifications = notifications

def create_speed_chart(df: pd.DataFrame):
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
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
    )
    return fig

def create_power_chart(df: pd.DataFrame):
    if df.empty or not all(col in df.columns for col in ["voltage_v", "current_a", "power_w"]):
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
        go.Scatter(x=df["timestamp"], y=df["voltage_v"], name="Voltage (V)", line=dict(color="#2ca02c", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["current_a"], name="Current (A)", line=dict(color="#d62728", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["power_w"], name="Power (W)", line=dict(color="#ff7f0e", width=2)),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=500,
        title_text="⚡ Electrical System Performance",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_imu_chart(df: pd.DataFrame):
    needed = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
    if df.empty or not all(col in df.columns for col in needed):
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
        subplot_titles=("🎯 Gyroscope Data (deg/s)", "📈 Accelerometer Data (m/s²)"),
        vertical_spacing=0.25,
    )
    colors_gyro = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, axis in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}", line=dict(color=colors_gyro[i], width=2)),
            row=1,
            col=1,
        )
    colors_accel = ["#f39c12", "#9b59b6", "#34495e"]
    for i, axis in enumerate(["accel_x", "accel_y", "accel_z"]):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}", line=dict(color=colors_accel[i], width=2)),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=600,
        title_text="🎮 IMU Sensor Data Analysis",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_imu_detail_chart(df: pd.DataFrame):
    needed = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
    if df.empty or not all(col in df.columns for col in needed):
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
        subplot_titles=("🌀 Gyro X", "🌀 Gyro Y", "🌀 Gyro Z", "📊 Accel X", "📊 Accel Y", "📊 Accel Z"),
        vertical_spacing=0.3,
        horizontal_spacing=0.1,
    )
    gyro_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    accel_colors = ["#f39c12", "#9b59b6", "#34495e"]
    for i, (axis, color) in enumerate(zip(["gyro_x", "gyro_y", "gyro_z"], gyro_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Gyro {axis[-1].upper()}", line=dict(color=color, width=2), showlegend=False),
            row=1,
            col=i + 1,
        )
    for i, (axis, color) in enumerate(zip(["accel_x", "accel_y", "accel_z"], accel_colors)):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[axis], name=f"Accel {axis[-1].upper()}", line=dict(color=color, width=2), showlegend=False),
            row=2,
            col=i + 1,
        )
    fig.update_layout(
        height=600,
        title_text="🎮 Detailed IMU Sensor Analysis",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_efficiency_chart(df: pd.DataFrame):
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
        color_continuous_scale="viridis",
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
    return fig

def create_gps_map_with_altitude(df: pd.DataFrame):
    if df.empty or not all(col in df.columns for col in ["latitude", "longitude"]):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    initial_rows = len(df)
    df_filtered = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    filtered_rows = len(df_filtered)
    if initial_rows > 0 and filtered_rows < initial_rows:
        st.warning(f"🛰️ Excluded {initial_rows - filtered_rows} invalid (0,0) coordinates.")
    df_valid = df_filtered.dropna(subset=["latitude", "longitude"])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates found after filtering",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("🛰️ Vehicle Track", "⛰️ Altitude Profile"),
        specs=[[{"type": "scattermapbox"}, {"type": "scatter"}]],
    )

    center_point = dict(lat=df_valid["latitude"].mean(), lon=df_valid["longitude"].mean())

    fig.add_trace(
        go.Scattermapbox(
            lat=df_valid["latitude"],
            lon=df_valid["longitude"],
            mode="markers+lines",
            marker=dict(
                size=8,
                color=df_valid["speed_ms"] if "speed_ms" in df_valid.columns else "#1f77b4",
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Speed (m/s)", x=0.65),
            ),
            line=dict(width=2, color="#1f77b4"),
            hovertemplate="Lat: %{lat}<br>Lon: %{lon}<br>Speed: %{marker.color:.1f} m/s<extra></extra>",
            name="Track",
        ),
        row=1,
        col=1,
    )

    if "altitude" in df.columns:
        altitude_data = df.dropna(subset=["altitude"])
        initial_alt_rows = len(altitude_data)
        altitude_data = altitude_data[altitude_data["altitude"] != 0]
        filtered_alt_rows = len(altitude_data)
        if initial_alt_rows > 0 and filtered_alt_rows < initial_alt_rows:
            st.warning(f"⛰️ Excluded {initial_alt_rows - filtered_alt_rows} rows with 0 altitude.")
        if not altitude_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=altitude_data["timestamp"],
                    y=altitude_data["altitude"],
                    mode="lines",
                    line=dict(color="#2ca02c", width=2),
                    name="Altitude",
                    hovertemplate="Time: %{x}<br>Altitude: %{y:.1f} m<extra></extra>",
                ),
                row=1,
                col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0], mode="text", text=["No valid altitude data"], textposition="middle center", showlegend=False
                ),
                row=1,
                col=2,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="text",
                text=["No altitude data available"],
                textposition="middle center",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title_text="🛰️ GPS Tracking and Altitude Analysis",
        height=500,
        showlegend=False,
        mapbox=dict(style="open-street-map", center=center_point, zoom=15, bearing=0, pitch=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=2)
    return fig

def get_available_columns(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["message_id", "uptime_seconds"]
    return [col for col in numeric_columns if col not in exclude_cols]

def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    x_col = chart_config.get("x_axis")
    y_col = chart_config.get("y_axis")
    chart_type = chart_config.get("chart_type", "line")
    title = chart_config.get("title", f"{y_col} vs {x_col}")

    if chart_type == "heatmap":
        numeric_cols = get_available_columns(df)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="🔥 Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
        else:
            return go.Figure().add_annotation(
                text="Need at least 2 numeric columns for heatmap",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
    else:
        if not y_col or y_col not in df.columns:
            return go.Figure().add_annotation(
                text="Invalid Y-axis column selection",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        if x_col not in df.columns:
            return go.Figure().add_annotation(
                text="Invalid X-axis column selection",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        try:
            if chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#ff7f0e"])
            elif chart_type == "bar":
                recent_df = df.tail(20)
                if recent_df.empty:
                    return go.Figure().add_annotation(
                        text="Not enough recent data for bar chart",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                    )
                fig = px.bar(recent_df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#2ca02c"])
            elif chart_type == "histogram":
                fig = px.histogram(df, x=y_col, title=f"Distribution of {y_col}", color_discrete_sequence=["#d62728"])
            else:
                fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#1f77b4"])
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"Error creating chart: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )

    fig.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def render_dynamic_charts_section(df: pd.DataFrame):
    if not st.session_state.chart_info_initialized:
        st.session_state.chart_info_text = """
        <div class="card">
            <h3 style="margin:0 0 .5rem 0;">🎯 Create Custom Charts</h3>
            <p>Click <strong>"Add Chart"</strong> to create custom visualizations with your preferred variables and chart types.</p>
        </div>
        """
        st.session_state.chart_types_grid = """
        <div class="card soft">
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem;">
                <div class="notice">📈 Line Chart — time series and trends</div>
                <div class="notice">🔵 Scatter — correlations</div>
                <div class="notice">📊 Bar — recent comparisons</div>
                <div class="notice">📉 Histogram — distributions</div>
                <div class="notice">🔥 Heatmap — correlations across variables</div>
            </div>
        </div>
        """
        st.session_state.chart_info_initialized = True

    st.markdown(st.session_state.chart_info_text, unsafe_allow_html=True)
    st.markdown(st.session_state.chart_types_grid, unsafe_allow_html=True)

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
                            index=["line", "scatter", "bar", "histogram", "heatmap"].index(
                                chart_config.get("chart_type", "line")
                            ),
                            key=f"type_{chart_config['id']}",
                        )
                        if new_type != chart_config.get("chart_type"):
                            st.session_state.dynamic_charts[i]["chart_type"] = new_type
                    with col3:
                        if chart_config.get("chart_type", "line") not in ["histogram", "heatmap"]:
                            x_options = ["timestamp"] + available_columns if "timestamp" in df.columns else available_columns
                            current_x = chart_config.get("x_axis")
                            if current_x not in x_options and x_options:
                                current_x = x_options[0]
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
                                st.empty()
                    with col4:
                        if chart_config.get("chart_type", "line") != "heatmap":
                            if available_columns:
                                current_y = chart_config.get("y_axis")
                                if current_y not in available_columns:
                                    current_y = available_columns[0]
                                new_y = st.selectbox(
                                    "Y-Axis",
                                    options=available_columns,
                                    index=available_columns.index(current_y) if current_y in available_columns else 0,
                                    key=f"y_{chart_config['id']}",
                                )
                                if new_y != chart_config.get("y_axis"):
                                    st.session_state.dynamic_charts[i]["y_axis"] = new_y
                            else:
                                st.empty()
                    with col5:
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"):
                            try:
                                idx_to_delete = next(
                                    (
                                        j
                                        for j, cfg in enumerate(st.session_state.dynamic_charts)
                                        if cfg["id"] == chart_config["id"]
                                    ),
                                    -1,
                                )
                                if idx_to_delete != -1:
                                    st.session_state.dynamic_charts.pop(idx_to_delete)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")

                    try:
                        fig = None
                        if chart_config.get("chart_type") == "heatmap":
                            fig = create_dynamic_chart(df, chart_config)
                        elif chart_config.get("y_axis") and (
                            chart_config.get("chart_type") == "histogram" or chart_config.get("x_axis")
                        ):
                            fig = create_dynamic_chart(df, chart_config)
                        else:
                            st.warning("Please select valid axes for the chosen chart type.")
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"dyn_plot_{chart_config['id']}",
                            )
                    except Exception as e:
                        st.error(f"Error rendering chart: {e}")

            except Exception as e:
                st.error(f"Error rendering chart configuration: {e}")

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
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect", use_container_width=True, key="btn_connect"):
                    if st.session_state.telemetry_manager:
                        st.session_state.telemetry_manager.disconnect()
                        time.sleep(0.5)
                    with st.spinner("Connecting..."):
                        st.session_state.telemetry_manager = EnhancedTelemetryManager()
                        supabase_connected = st.session_state.telemetry_manager.connect_supabase()
                        realtime_connected = False
                        if ABLY_AVAILABLE:
                            realtime_connected = st.session_state.telemetry_manager.connect_realtime()
                        if supabase_connected and realtime_connected:
                            st.success("✅ Connected!")
                        elif supabase_connected:
                            st.warning("⚠️ Supabase only connected (Ably not available or failed)")
                        else:
                            st.error("❌ Failed to connect to any service!")
                    st.rerun()
            with col2:
                if st.button("🛑 Disconnect", use_container_width=True, key="btn_disconnect"):
                    if st.session_state.telemetry_manager:
                        st.session_state.telemetry_manager.disconnect()
                        st.session_state.telemetry_manager = None
                    st.info("🛑 Disconnected")
                    st.rerun()

            if st.session_state.telemetry_manager:
                stats = st.session_state.telemetry_manager.get_stats()
                if st.session_state.telemetry_manager.is_connected:
                    st.markdown('<div class="status-indicator status-connected">✅ Real-time Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-indicator status-disconnected">❌ Real-time Disconnected</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("📨 Messages", stats["messages_received"])
                    st.metric("🔌 Attempts", stats["connection_attempts"])
                with c2:
                    st.metric("❌ Errors", stats["errors"])
                    if stats["last_message_time"]:
                        time_since = (datetime.now() - stats["last_message_time"]).total_seconds()
                        st.metric("⏱️ Last Msg", f"{time_since:.0f}s ago")
                    else:
                        st.metric("⏱️ Last Msg", "Never")
                if stats["last_error"]:
                    st.error(f"⚠️ {stats['last_error'][:40]}...")

            st.divider()
            st.subheader("⚙️ Settings")
            auto_refresh_key = f"auto_refresh_{id(st.session_state)}"
            new_auto_refresh = st.checkbox(
                "🔄 Auto Refresh",
                value=st.session_state.auto_refresh,
                help="Automatically refresh data from real-time stream",
                key=auto_refresh_key,
            )
            if new_auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = new_auto_refresh
            if st.session_state.auto_refresh:
                refresh_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                current_index = refresh_options.index(3) if 3 in refresh_options else 2
                refresh_interval = st.selectbox(
                    "Refresh Rate (seconds)",
                    options=refresh_options,
                    index=current_index,
                    key=f"refresh_rate_{id(st.session_state)}",
                )
            else:
                refresh_interval = 3
            st.session_state.refresh_interval = refresh_interval

        else: # Historical data mode
            st.markdown('<div class="status-indicator status-historical">📚 Historical Mode</div>', unsafe_allow_html=True)
            if not st.session_state.telemetry_manager:
                st.session_state.telemetry_manager = EnhancedTelemetryManager()
                st.session_state.telemetry_manager.connect_supabase()
            if st.button("🔄 Refresh Sessions", use_container_width=True, key="btn_refresh_sessions"):
                with st.spinner("Loading sessions..."):
                    st.session_state.historical_sessions = st.session_state.telemetry_manager.get_historical_sessions()
                st.rerun()
            if st.session_state.historical_sessions:
                session_options = []
                for session in st.session_state.historical_sessions:
                    session_options.append(
                        f"{session['session_id'][:8]}... - {session['start_time'].strftime('%Y-%m-%d %H:%M')} ({session['record_count']:,} records)"
                    )
                selected_session_idx = st.selectbox(
                    "📋 Select Session",
                    options=range(len(session_options)),
                    format_func=lambda x: session_options[x],
                    key="session_selector",
                    index=0,
                )
                if selected_session_idx is not None:
                    selected_session = st.session_state.historical_sessions[selected_session_idx]
                    if (
                        st.session_state.selected_session is None
                        or st.session_state.selected_session["session_id"] != selected_session["session_id"]
                        or st.session_state.telemetry_data.empty
                    ):
                        st.session_state.selected_session = selected_session
                        st.session_state.is_viewing_historical = True
                        if selected_session["record_count"] > 10000:
                            st.info(
                                f"📊 Loading {selected_session['record_count']:,} records... This may take a moment due to pagination."
                            )
                        with st.spinner(f"Loading data for session {selected_session['session_id'][:8]}..."):
                            historical_df = st.session_state.telemetry_manager.get_historical_data(
                                selected_session["session_id"]
                            )
                            st.session_state.telemetry_data = historical_df
                            st.session_state.last_update = datetime.now()
                        if not historical_df.empty:
                            st.success(f"✅ Loaded {len(historical_df):,} data points")
                        st.rerun()
            else:
                st.info("Click 'Refresh Sessions' to load available sessions from Supabase.")

        st.info(f"📡 Channel: {DASHBOARD_CHANNEL_NAME}")
        st.info(f"🔢 Max records per session: {MAX_DATAPOINTS_PER_SESSION:,}")

    # Main content area
    df = st.session_state.telemetry_data.copy()
    new_messages_count = 0

    # Data ingestion logic
    if st.session_state.data_source_mode == "realtime_session":
        if st.session_state.telemetry_manager and st.session_state.telemetry_manager.is_connected:
            new_messages = st.session_state.telemetry_manager.get_realtime_messages()
            current_session_data_from_supabase = pd.DataFrame()
            if new_messages and "session_id" in new_messages[0]:
                current_session_id = new_messages[0]["session_id"]
                if st.session_state.current_session_id != current_session_id or st.session_state.telemetry_data.empty:
                    st.session_state.current_session_id = current_session_id
                    with st.spinner(f"Loading current session data for {current_session_id[:8]}..."):
                        current_session_data_from_supabase = st.session_state.telemetry_manager.get_current_session_data(current_session_id)
                    if not current_session_data_from_supabase.empty:
                        st.success(f"✅ Loaded {len(current_session_data_from_supabase):,} historical points for current session")
            if new_messages or not current_session_data_from_supabase.empty:
                merged_data = merge_telemetry_data(
                    new_messages,
                    current_session_data_from_supabase,
                    st.session_state.telemetry_data,
                )
                if not merged_data.empty:
                    new_messages_count = len(new_messages) if new_messages else 0
                    st.session_state.telemetry_data = merged_data
                    st.session_state.last_update = datetime.now()
        st.session_state.is_viewing_historical = False
    elif st.session_state.data_source_mode == "historical":
        st.session_state.is_viewing_historical = True

    df = st.session_state.telemetry_data.copy()

    if st.session_state.is_viewing_historical and st.session_state.selected_session:
        st.markdown('<div class="notice">📚 Viewing Historical Data - No auto-refresh active</div>', unsafe_allow_html=True)
        render_session_info(st.session_state.selected_session)

    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.data_source_mode == "realtime_session":
                st.info(
                    "**Getting Started (Real-time):**\n"
                    "1. Ensure the bridge (your data sending script) is running\n"
                    "2. Click 'Connect' in the sidebar to start receiving data\n"
                    "3. Large sessions are automatically paginated for optimal performance"
                )
            else:
                st.info(
                    "**Getting Started (Historical):**\n"
                    "1. Click 'Refresh Sessions' in the sidebar to load available sessions\n"
                    "2. Select a session and its data will load automatically\n"
                    "3. Large datasets use pagination to load all data points"
                )
        with col2:
            with st.expander("🔍 Debug Information"):
                debug_info = {
                    "Data Source Mode": st.session_state.data_source_mode,
                    "Is Viewing Historical": st.session_state.is_viewing_historical,
                    "Selected Session ID": st.session_state.selected_session["session_id"][:8] + "..." if st.session_state.selected_session else None,
                    "Current Real-time Session ID": st.session_state.current_session_id,
                    "Num Historical Sessions": len(st.session_state.historical_sessions),
                    "Telemetry Data Points (in memory)": len(st.session_state.telemetry_data),
                }
                if st.session_state.telemetry_manager:
                    stats = st.session_state.telemetry_manager.get_stats()
                    debug_info.update({
                        "Ably Connected": st.session_state.telemetry_manager.is_connected,
                        "Messages Received": stats["messages_received"],
                        "Connection Errors": stats["errors"],
                        "Total Pagination Requests": stats["pagination_stats"]["total_requests"],
                        "Total Rows Fetched": stats["pagination_stats"]["total_rows_fetched"],
                    })
                st.json(debug_info)
        return

    analyze_data_quality(df, is_realtime=(st.session_state.data_source_mode == "realtime_session"))
    if st.session_state.data_quality_notifications:
        for msg in st.session_state.data_quality_notifications:
            if "🚨" in msg:
                st.error(msg, icon="🚨")
            else:
                st.warning(msg, icon="⚠️")

    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        st.info("📚 Historical" if st.session_state.is_viewing_historical else "🔴 Real-time")
    with col2:
        st.info(f"📊 **{len(df):,}** data points")
    with col3:
        st.info(f"⏰ Last update: **{st.session_state.last_update.strftime('%H:%M:%S')}**")
    with col4:
        if st.session_state.data_source_mode == "realtime_session" and new_messages_count > 0:
            st.success(f"📨 +{new_messages_count}")

    if len(df) > 10000:
        st.markdown(f'<div class="notice"><strong>📊 Large Dataset Loaded:</strong> {len(df):,} data points successfully retrieved using pagination</div>', unsafe_allow_html=True)

    kpis = calculate_kpis(df)

    st.subheader("📈 Dashboard")
    tab_names = ["📊 Overview", "🚗 Speed", "⚡ Power", "🎮 IMU", "🎮 IMU Detail", "⚡ Efficiency", "🛰️ GPS", "📈 Custom", "📃 Data"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_overview_tab(kpis, key_prefix="overview_tab")
    with tabs[1]:
        render_kpi_header(kpis, key_prefix="speed_tab")
        st.plotly_chart(create_speed_chart(df), use_container_width=True, key="speed_chart_main")
    with tabs[2]:
        render_kpi_header(kpis, key_prefix="power_tab")
        st.plotly_chart(create_power_chart(df), use_container_width=True, key="power_chart_main")
    with tabs[3]:
        render_kpi_header(kpis, key_prefix="imu_tab")
        st.plotly_chart(create_imu_chart(df), use_container_width=True, key="imu_chart_main")
    with tabs[4]:
        render_kpi_header(kpis, key_prefix="imu_detail_tab")
        st.plotly_chart(create_imu_detail_chart(df), use_container_width=True, key="imu_detail_chart_main")
    with tabs[5]:
        render_kpi_header(kpis, key_prefix="efficiency_tab")
        st.plotly_chart(create_efficiency_chart(df), use_container_width=True, key="efficiency_chart_main")
    with tabs[6]:
        render_kpi_header(kpis, key_prefix="gps_tab")
        st.plotly_chart(create_gps_map_with_altitude(df), use_container_width=True, key="gps_map_main")
    with tabs[7]:
        render_kpi_header(kpis, key_prefix="custom_tab")
        render_dynamic_charts_section(df)
    with tabs[8]:
        render_kpi_header(kpis, key_prefix="data_tab")
        st.subheader("📃 Raw Telemetry Data")
        st.info(f"ℹ️ Showing last 100 of {len(df):,} data points.")
        st.dataframe(df.tail(100), use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        with c1:
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Full CSV ({len(df):,} rows)",
                data=csv,
                file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if len(df) > 1000:
                sample_df = df.sample(n=min(1000, len(df)), random_state=42)
                sample_csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample CSV (1000 rows)",
                    data=sample_csv,
                    file_name=f"telemetry_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with st.expander("📊 Dataset Statistics"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Rows", f"{len(df):,}")
                st.metric("Columns", len(df.columns))
            with c2:
                if "timestamp" in df.columns and len(df) > 1:
                    try:
                        ts_series = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
                        if len(ts_series) > 1:
                            time_span = ts_series.max() - ts_series.min()
                            st.metric("Time Span", str(time_span).split(".")[0])
                            if time_span.total_seconds() > 0:
                                data_rate = len(df) / time_span.total_seconds()
                                st.metric("Data Rate", f"{data_rate:.2f} Hz")
                    except Exception:
                        st.metric("Time Span", "Error")
                        st.metric("Data Rate", "Error")
            with c3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")
                if "data_source" in df.columns:
                    st.write("**Data Sources:**")
                    st.dataframe(df["data_source"].value_counts())

    if st.session_state.data_source_mode == "realtime_session" and st.session_state.auto_refresh:
        if AUTOREFRESH_AVAILABLE:
            st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="auto_refresh_component")
        else:
            st.warning("🔄 To enable smooth auto-refresh install: `pip install streamlit-autorefresh`")

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--muted); padding: 1rem;'>"
        "<p>Shell Eco-marathon Telemetry Dashboard</p>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
