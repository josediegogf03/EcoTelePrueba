import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import time
import logging
import asyncio
from collections import deque

# Try to import Ably REST, fall back if not installed
try:
    from ably import AblyRest, AblyException
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    AblyRest = None
    AblyException = Exception

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Ably Configuration ---
ABLY_API_KEY_FALLBACK = (
    "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
)
ABLY_API_KEY = os.environ.get("ABLY_API_KEY", ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"

# --- Constants ---
MAX_DATAPOINTS_IN_DASHBOARD = 500
PLACEHOLDER_COLS = [
    "timestamp",
    "speed_ms",
    "voltage_v",
    "current_a",
    "power_w",
    "energy_j",
    "distance_m",
    "latitude",
    "longitude",
]

# --- Page Configuration ---
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard V5",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
}
.error-card {
    background-color: #ff6b35;
    border: 2px solid #e55100;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: white;
}
.warning-card {
    background-color: #d32f2f;
    border: 2px solid #b71c1c;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: white;
}
.info-card {
    background-color: #1976d2;
    border: 2px solid #0d47a1;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: white;
}
.info-card h4, .info-card h5 {
    color: #e3f2fd;
}
.info-card ul {
    color: #f3e5f5;
}
.warning-card h3 {
    color: #ffebee;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Initialize Session State (Proper Streamlit Pattern) ---
if "connection_error" not in st.session_state:
    st.session_state.connection_error = None

if "use_mock_data" not in st.session_state:
    st.session_state.use_mock_data = False

if "ably_rest" not in st.session_state:
    st.session_state.ably_rest = None

if "ably_channel" not in st.session_state:
    st.session_state.ably_channel = None

if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = None

if "telemetry_data_deque" not in st.session_state:
    st.session_state.telemetry_data_deque = deque(
        maxlen=MAX_DATAPOINTS_IN_DASHBOARD
    )

# --- Initialize Ably REST Client once ---
if (
    ABLY_AVAILABLE
    and st.session_state.ably_rest is None
    and st.session_state.connection_error is None
):
    try:
        rest = AblyRest(key=ABLY_API_KEY)
        ch = rest.channels.get(TELEMETRY_CHANNEL_NAME)
        st.session_state.ably_rest = rest
        st.session_state.ably_channel = ch
        logging.info("Initialized AblyRest client and channel.")
    except AblyException as e:
        st.session_state.connection_error = f"Ably REST init failed: {e}"
        logging.error(st.session_state.connection_error)

# --- Async Helper ---
async def fetch_history_async(channel, limit):
    """Async wrapper for channel history"""
    try:
        history = await channel.history(direction="forwards", limit=limit)
        return history
    except Exception as e:
        raise AblyException(f"Failed to fetch history: {e}")

def fetch_ably_data():
    """Fetch data from Ably channel, handling both sync and async cases"""
    if not st.session_state.ably_channel:
        return
    
    try:
        # Try async approach first
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop
