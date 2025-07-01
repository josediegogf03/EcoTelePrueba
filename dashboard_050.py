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

# Disable tracemalloc warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Ably import with error handling
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("❌ Ably library not available. Please install: pip install ably")
    st.stop()

# --- Optimized logging setup ---
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger('TelemetrySubscriber')
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

setup_terminal_logging()

# Configuration
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATAPOINTS = 50000

# Page configuration with optimized settings
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optimized CSS with theme compatibility and scroll prevention
st.markdown("""
<style>
    /* Theme-compatible colors */
    :root {
        --primary-color: #ff4b4b;
        --success-color: #00d4aa;
        --warning-color: #ffab00;
        --error-color: #ff4b4b;
        --info-color: #00a0ff;
        --text-primary: var(--text-color, #262730);
        --text-secondary: #7f8c8d;
        --background-primary: var(--background-color, #ffffff);
        --background-secondary: var(--secondary-background-color, #f0f2f6);
        --border-color: #e6e6e6;
    }
    
    [data-theme="dark"] {
        --text-primary: #fafafa;
        --text-secondary: #a6a6a6;
        --background-primary: #0e1117;
        --background-secondary: #262730;
        --border-color: #3a3a3a;
    }
    
    /* Fixed header for better navigation */
    .fixed-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: var(--background-primary);
        padding: 1rem 0;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 2.2rem;
        color: var(--primary-color);
        text-align: center;
        margin: 0;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Connection status - moved to top for visibility */
    .connection-banner {
        position: sticky;
        top: 0;
        z-index: 999;
        padding: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .status-connected {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
    }
    
    .status-disconnected {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }
    
    .status-connecting {
        background: linear-gradient(135deg, #ffd93d 0%, #ff6b35 100%);
        color: #2d3436;
        box-shadow: 0 2px 8px rgba(255, 217, 61, 0.3);
    }
    
    /* Optimized KPI cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: var(--background-secondary);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid var(--primary-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color), transparent);
    }
    
    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chart containers with fixed positioning */
    .chart-section {
        position: relative;
        margin-bottom: 2rem;
    }
    
    .chart-container {
        background: var(--background-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
    }
    
    /* Dynamic charts section */
    .dynamic-charts-header {
        background: linear-gradient(135deg, var(--primary-color), #e17055);
        color: white;
        padding: 1.5rem;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
    }
    
    .dynamic-charts-body {
        background: var(--background-secondary);
        border-radius: 0 0 12px 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        border-top: none;
    }
    
    .chart-config-card {
        background: var(--background-primary);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 2px dashed var(--border-color);
        transition: border-color 0.3s ease;
    }
    
    .chart-config-card:hover {
        border-color: var(--primary-color);
    }
    
    /* Instructions styling */
    .instructions-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .instructions-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chart-types-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .chart-type-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    .chart-type-name {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Prevent scroll jumping */
    .main .block-container {
        scroll-behavior: smooth;
        max-width: none;
    }
    
    /* Fixed positioning elements */
    .element-container {
        scroll-margin-top: 120px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
        
        .main-title {
            font-size: 1.8rem;
        }
        
        .chart-types-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Loading animations */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Smooth transitions */
    * {
        transition: color 0.3s ease, background-color 0.3s ease, border-color 0.3s ease;
    }
</style>

<script>
// Optimized scroll position preservation
(function() {
    const SCROLL_KEY = 'streamlit_scroll_position';
    
    function saveScrollPosition() {
        sessionStorage.setItem(SCROLL_KEY, JSON.stringify({
            x: window.scrollX,
            y: window.scrollY,
            timestamp: Date.now()
        }));
    }
    
    function restoreScrollPosition() {
        try {
            const stored = sessionStorage.getItem(SCROLL_KEY);
            if (stored) {
                const position = JSON.parse(stored);
                // Only restore if position was saved recently (within 5 seconds)
                if (Date.now() - position.timestamp < 5000) {
                    setTimeout(() => {
                        window.scrollTo({
                            left: position.x,
                            top: position.y,
                            behavior: 'auto'
                        });
                    }, 100);
                }
            }
        } catch (e) {
            console.warn('Failed to restore scroll position:', e);
        }
    }
    
    // Save scroll position before page unload
    window.addEventListener('beforeunload', saveScrollPosition);
    
    // Restore on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', restoreScrollPosition);
    } else {
        restoreScrollPosition();
    }
    
    // Handle Streamlit reruns
    setTimeout(restoreScrollPosition, 500);
})();
</script>
""", unsafe_allow_html=True)

# Optimized TelemetrySubscriber class (minimal changes for reliability)
class TelemetrySubscriber:
    def __init__(self):
        self.ably_client = None
        self.channel = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.connection_thread = None
        self.stats = {
            'messages_received': 0,
            'last_message_time': None,
            'connection_attempts': 0,
            'errors': 0,
            'last_error': None
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_run = False
        self.logger = logging.getLogger('TelemetrySubscriber')
    
    def connect(self) -> bool:
        """Connect to Ably and start receiving messages"""
        try:
            with self._lock:
                self.stats['connection_attempts'] += 1
            
            self.logger.info("🔄 Starting connection to Ably...")
            
            if self._should_run:
                self.disconnect()
            
            self._stop_event.clear()
            self._should_run = True
            
            self.connection_thread = threading.Thread(target=self._connection_worker, daemon=True)
            self.connection_thread.start()
            
            time.sleep(3)
            return self.is_connected
            
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
            self.is_connected = False
            return False
    
    def _connection_worker(self):
        """Worker thread to handle Ably connection"""
        try:
            self.logger.info("🔄 Connection worker starting...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_connection_handler())
        except Exception as e:
            self.logger.error(f"💥 Connection worker error: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
            self.is_connected = False
        finally:
            self.logger.info("🛑 Connection worker ended")
    
    async def _async_connection_handler(self):
        """Handle Ably connection asynchronously"""
        try:
            self.logger.info("🔗 Creating Ably client...")
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
                    self.stats['errors'] += 1
                    self.stats['last_error'] = f"Connection failed: {state_change}"
            
            self.ably_client.connection.on('connected', on_connected)
            self.ably_client.connection.on('disconnected', on_disconnected)
            self.ably_client.connection.on('failed', on_failed)
            self.ably_client.connection.on('suspended', on_disconnected)
            
            self.logger.info("⏳ Waiting for connection...")
            await self.ably_client.connection.once_async('connected')
            
            self.logger.info(f"📡 Getting channel: {CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(CHANNEL_NAME)
            
            self.logger.info("📨 Subscribing to messages...")
            await self.channel.subscribe('telemetry_update', self._on_message_received)
            
            self.logger.info("✅ Successfully subscribed to messages!")
            
            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)
                
                if hasattr(self.ably_client.connection, 'state'):
                    state = self.ably_client.connection.state
                    if state not in ['connected']:
                        self.logger.warning(f"⚠️ Connection state: {state}")
                        if state in ['failed', 'suspended', 'disconnected']:
                            self.is_connected = False
                            break
            
            self.logger.info("🔚 Connection loop ended")
            
        except Exception as e:
            self.logger.error(f"💥 Async connection error: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
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
                        self.stats['errors'] += 1
                        self.stats['last_error'] = f"JSON decode error: {e}"
                    return
            
            if not isinstance(data, dict):
                self.logger.warning(f"⚠️ Invalid data type: {type(data)}")
                with self._lock:
                    self.stats['errors'] += 1
                    self.stats['last_error'] = f"Invalid data type: {type(data)}"
                return
            
            self.logger.debug(f"📊 Data keys: {list(data.keys())}")
            
            with self._lock:
                if self.message_queue.qsize() > 100:
                    try:
                        while self.message_queue.qsize() > 50:
                            self.message_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.message_queue.put(data)
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = datetime.now()
                
                self.logger.debug(f"✅ Message queued. Total: {self.stats['messages_received']}")
            
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = f"Message error: {e}"
    
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
                self.stats['errors'] += 1
                self.stats['last_error'] = f"Disconnect error: {e}"
        finally:
            self.ably_client = None
            self.channel = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self._lock:
            return self.stats.copy()

# Optimized session state initialization
def initialize_session_state():
    """Initialize Streamlit session state with optimized defaults"""
    defaults = {
        'subscriber': None,
        'telemetry_data': pd.DataFrame(),
        'last_update': datetime.now(),
        'auto_refresh': True,
        'dynamic_charts': [],
        'active_tab': 0,
        'is_auto_refresh': False,
        'scroll_position': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Optimized KPI calculation with error handling
@st.cache_data(ttl=2)  # Cache for 2 seconds to improve performance
def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators with improved error handling"""
    default_kpis = {
        'total_energy_mj': 0.0,
        'max_speed_ms': 0.0,
        'avg_speed_ms': 0.0,
        'total_distance_km': 0.0,
        'avg_power_w': 0.0,
        'efficiency_km_per_mj': 0.0,
        'max_acceleration': 0.0,
        'avg_gyro_magnitude': 0.0
    }
    
    if df.empty:
        return default_kpis
    
    try:
        # Ensure numeric columns with error handling
        numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w', 'total_acceleration', 'gyro_x', 'gyro_y', 'gyro_z']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Safe calculations with fallbacks
        total_energy = df['energy_j'].iloc[-1] / 1_000_000 if 'energy_j' in df.columns and len(df) > 0 else 0
        max_speed = df['speed_ms'].max() if 'speed_ms' in df.columns and not df['speed_ms'].empty else 0
        avg_speed = df['speed_ms'].mean() if 'speed_ms' in df.columns and not df['speed_ms'].empty else 0
        total_distance = df['distance_m'].iloc[-1] / 1000 if 'distance_m' in df.columns and len(df) > 0 else 0
        avg_power = df['power_w'].mean() if 'power_w' in df.columns and not df['power_w'].empty else 0
        efficiency = total_distance / total_energy if total_energy > 0 else 0
        
        # IMU-based KPIs with error handling
        max_acceleration = df['total_acceleration'].max() if 'total_acceleration' in df.columns and not df['total_acceleration'].empty else 0
        
        if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            try:
                gyro_magnitude = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
                avg_gyro_magnitude = gyro_magnitude.mean() if not gyro_magnitude.empty else 0
            except Exception:
                avg_gyro_magnitude = 0
        else:
            avg_gyro_magnitude = 0
        
        return {
            'total_energy_mj': float(total_energy) if not np.isnan(total_energy) else 0.0,
            'max_speed_ms': float(max_speed) if not np.isnan(max_speed) else 0.0,
            'avg_speed_ms': float(avg_speed) if not np.isnan(avg_speed) else 0.0,
            'total_distance_km': float(total_distance) if not np.isnan(total_distance) else 0.0,
            'avg_power_w': float(avg_power) if not np.isnan(avg_power) else 0.0,
            'efficiency_km_per_mj': float(efficiency) if not np.isnan(efficiency) else 0.0,
            'max_acceleration': float(max_acceleration) if not np.isnan(max_acceleration) else 0.0,
            'avg_gyro_magnitude': float(avg_gyro_magnitude) if not np.isnan(avg_gyro_magnitude) else 0.0
        }
    except Exception as e:
        st.warning(f"Error calculating KPIs: {e}")
        return default_kpis

# Optimized KPI dashboard with better layout
def render_kpi_dashboard(kpis: Dict[str, float]):
    """Render optimized KPI dashboard with better layout and theme compatibility"""
    
    kpi_data = [
        ("🛣️", "Distance", f"{kpis['total_distance_km']:.2f}", "km"),
        ("⚡", "Max Speed", f"{kpis['max_speed_ms']:.1f}", "m/s"),
        ("🏃", "Avg Speed", f"{kpis['avg_speed_ms']:.1f}", "m/s"),
        ("🔋", "Energy", f"{kpis['total_energy_mj']:.2f}", "MJ"),
        ("💡", "Avg Power", f"{kpis['avg_power_w']:.1f}", "W"),
        ("♻️", "Efficiency", f"{kpis['efficiency_km_per_mj']:.2f}", "km/MJ"),
        ("📈", "Max Accel", f"{kpis['max_acceleration']:.2f}", "m/s²"),
        ("🎯", "Avg Gyro", f"{kpis['avg_gyro_magnitude']:.2f}", "°/s")
    ]
    
    # Create KPI grid HTML
    kpi_html = '<div class="kpi-grid">'
    
    for icon, label, value, unit in kpi_data:
        kpi_html += f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value} <span style="font-size: 0.8em; opacity: 0.7;">{unit}</span></div>
            <div class="kpi-label">{label}</div>
        </div>
        """
    
    kpi_html += '</div>'
    
    st.markdown(kpi_html, unsafe_allow_html=True)

# Optimized chart creation functions (keeping existing logic but with caching)
@st.cache_data(ttl=2)
def create_speed_chart(df: pd.DataFrame):
    """Create speed over time chart with caching"""
    if df.empty or 'speed_ms' not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.line(
        df, x='timestamp', y='speed_ms',
        title='🚀 Vehicle Speed Over Time',
        labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'},
        color_discrete_sequence=['#ff4b4b']
    )
    fig.update_layout(
        height=400,
        template="plotly_white",
        title_font_size=16,
        title_font_color='#ff4b4b'
    )
    return fig

@st.cache_data(ttl=2)
def create_power_chart(df: pd.DataFrame):
    """Create power system chart with caching"""
    if df.empty or not all(col in df.columns for col in ['voltage_v', 'current_a', 'power_w']):
        return go.Figure().add_annotation(
            text="No power data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=('⚡ Voltage & Current', '🔋 Power'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['voltage_v'], 
                  name='Voltage (V)', line=dict(color='#00d4aa')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['current_a'], 
                  name='Current (A)', line=dict(color='#ff6b6b')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_w'], 
                  name='Power (W)', line=dict(color='#74b9ff')), 
        row=2, col=1
    )
    
    fig.update_layout(
        height=500, 
        title_text="🔌 Electrical System Performance",
        template="plotly_white",
        title_font_size=16,
        title_font_color='#ff4b4b'
    )
    return fig

@st.cache_data(ttl=2)
def create_imu_chart(df: pd.DataFrame):
    """Create IMU chart with caching"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🌀 Gyroscope Data (deg/s)', '📊 Accelerometer Data (m/s²)'),
        vertical_spacing=0.25
    )
    
    # Gyroscope data
    colors = ['#ff6b6b', '#00d4aa', '#74b9ff']
    gyro_labels = ['Gyro X', 'Gyro Y', 'Gyro Z']
    gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
    
    for i, (col, label, color) in enumerate(zip(gyro_cols, gyro_labels, colors)):
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[col], 
                      name=label, line=dict(color=color)), 
            row=1, col=1
        )
    
    # Accelerometer data
    accel_labels = ['Accel X', 'Accel Y', 'Accel Z']
    accel_cols = ['accel_x', 'accel_y', 'accel_z']
    accel_colors = ['#fdcb6e', '#a29bfe', '#fd79a8']
    
    for i, (col, label, color) in enumerate(zip(accel_cols, accel_labels, accel_colors)):
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[col], 
                      name=label, line=dict(color=color)), 
            row=2, col=1
        )
      
    fig.update_layout(
        height=600, 
        title_text="🎯 IMU Sensor Data",
        template="plotly_white",
        title_font_size=16,
        title_font_color='#ff4b4b'
    )
    return fig

@st.cache_data(ttl=2)
def create_imu_chart_2(df: pd.DataFrame):
    """Create detailed IMU chart with individual subplots"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig2 = make_subplots(
        rows=2, cols=3,
        subplot_titles=('🌀 Gyroscope (X)', '🌀 Gyroscope (Y)', '🌀 Gyroscope (Z)', 
                       '📊 Accelerometer (X)', '📊 Accelerometer (Y)', '📊 Accelerometer (Z)'),
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    # Colors for different axes
    colors = {
        'gyro_x': '#e17055', 'gyro_y': '#00b894', 'gyro_z': '#0984e3',
        'accel_x': '#fdcb6e', 'accel_y': '#a29bfe', 'accel_z': '#fd79a8'
    }

    # Gyroscope data
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['gyro_x'], 
                            name='Gyro X', line=dict(color=colors['gyro_x'])), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['gyro_y'], 
                            name='Gyro Y', line=dict(color=colors['gyro_y'])), row=1, col=2)
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['gyro_z'], 
                            name='Gyro Z', line=dict(color=colors['gyro_z'])), row=1, col=3)
    
    # Accelerometer data
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['accel_x'], 
                            name='Accel X', line=dict(color=colors['accel_x'])), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['accel_y'], 
                            name='Accel Y', line=dict(color=colors['accel_y'])), row=2, col=2)
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['accel_z'], 
                            name='Accel Z', line=dict(color=colors['accel_z'])), row=2, col=3)

    fig2.update_layout(
        height=600, 
        title_text="🎯 Detailed IMU Sensor Analysis",
        template="plotly_white",
        title_font_size=16,
        title_font_color='#ff4b4b',
        showlegend=False
    )
    return fig2

@st.cache_data(ttl=2)
def create_efficiency_chart(df: pd.DataFrame):
    """Create efficiency analysis chart"""
    if df.empty or not all(col in df.columns for col in ['speed_ms', 'power_w']):
        return go.Figure().add_annotation(
            text="No efficiency data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.scatter(
        df, x='speed_ms', y='power_w',
        color='voltage_v' if 'voltage_v' in df.columns else None,
        title='⚡ Efficiency Analysis: Speed vs Power Consumption',
        labels={'speed_ms': 'Speed (m/s)', 'power_w': 'Power (W)'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=400,
        template="plotly_white",
        title_font_size=16,
        title_font_color='#ff4b4b'
    )
    return fig

@st.cache_data(ttl=2)
def create_gps_map(df: pd.DataFrame):
    """Create GPS tracking map"""
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude']):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    center_point = dict(
        lat=df_valid['latitude'].mean(),
        lon=df_valid['longitude'].mean()
    )
    
    fig = px.scatter_mapbox(
        df_valid, 
        lat='latitude', 
        lon='longitude',
        color='speed_ms' if 'speed_ms' in df_valid.columns else None,
        size='power_w' if 'power_w' in df_valid.columns else None,
        hover_data=['speed_ms', 'power_w', 'voltage_v'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w', 'voltage_v']) else None,
        mapbox_style='open-street-map',
        title='🗺️ Vehicle Track and Performance',
        height=400,
        zoom=12,
        center=center_point,
        color_continuous_scale='Plasma'
    )
    
    return fig

# Optimized dynamic chart functions
def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns available for plotting"""
    if df.empty:
        return []
    
    try:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['message_id', 'uptime_seconds']
        return [col for col in numeric_columns if col not in exclude_cols]
    except Exception:
        return []

@st.cache_data(ttl=2)
def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create a dynamic chart based on user configuration with improved error handling"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    try:
        x_col = chart_config.get('x_axis')
        y_col = chart_config.get('y_axis')
        chart_type = chart_config.get('chart_type', 'line')
        title = chart_config.get('title', f'{y_col} vs {x_col}')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return go.Figure().add_annotation(
                text="Invalid column selection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Color schemes for different chart types
        colors = ['#ff4b4b', '#00d4aa', '#74b9ff', '#fdcb6e', '#a29bfe']
        
        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=colors)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=colors)
        elif chart_type == 'bar':
            recent_df = df.tail(20)
            fig = px.bar(recent_df, x=x_col, y=y_col, title=title, color_discrete_sequence=colors)
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=y_col, title=f'Distribution of {y_col}', color_discrete_sequence=colors)
        elif chart_type == 'heatmap':
            # Create correlation heatmap or pivot table based on data
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                numeric_df = df.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(numeric_df, title=f'Correlation Heatmap', color_continuous_scale='RdBu')
            else:
                fig = go.Figure().add_annotation(
                    text="Need at least 2 numeric columns for heatmap",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=colors)
        
        fig.update_layout(
            height=400,
            template="plotly_white",
            title_font_size=14,
            title_font_color='#ff4b4b'
        )
        return fig
    
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

# Optimized dynamic charts section
def render_dynamic_charts_section(df: pd.DataFrame):
    """Render enhanced dynamic charts section with improved instructions"""
    
    # Instructions card
    st.markdown("""
    <div class="instructions-card">
        <div class="instructions-title">
            🎯 Create Custom Charts
        </div>
        <p>Click "Add Chart" to create custom visualizations with your preferred variables and chart types.</p>
        
        <div class="chart-types-grid">
            <div class="chart-type-item">
                <div class="chart-type-name">📈 Line</div>
                <div>Great for time series data and trends</div>
            </div>
            <div class="chart-type-item">
                <div class="chart-type-name">📊 Scatter</div>
                <div>Perfect for correlation analysis</div>
            </div>
            <div class="chart-type-item">
                <div class="chart-type-name">📋 Bar</div>
                <div>Good for comparing recent values</div>
            </div>
            <div class="chart-type-item">
                <div class="chart-type-name">📊 Histogram</div>
                <div>Shows data distribution patterns</div>
            </div>
            <div class="chart-type-item">
                <div class="chart-type-name">🔥 Heatmap</div>
                <div>Reveals correlations and patterns</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available columns
    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []
    
    if not available_columns:
        st.warning("⚠️ No numeric data available for creating charts.")
        return
    
    # Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➕ Add Custom Chart", key="add_chart_btn", use_container_width=True):
            try:
                new_chart = {
                    'id': str(uuid.uuid4()),
                    'title': f'Chart {len(st.session_state.dynamic_charts) + 1}',
                    'chart_type': 'line',
                    'x_axis': 'timestamp' if 'timestamp' in df.columns else available_columns[0],
                    'y_axis': available_columns[0] if available_columns else None
                }
                st.session_state.dynamic_charts.append(new_chart)
                st.rerun()
            except Exception as e:
                st.error(f"Error adding chart: {e}")
    
    # Display charts
    if st.session_state.dynamic_charts:
        st.markdown(f"**📊 {len(st.session_state.dynamic_charts)} Custom Chart(s)**")
        
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            try:
                with st.container():
                    st.markdown('<div class="chart-config-card">', unsafe_allow_html=True)
                    
                    # Chart configuration
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    
                    with col1:
                        new_title = st.text_input(
                            "Title", 
                            value=chart_config.get('title', 'New Chart'),
                            key=f"title_{chart_config['id']}"
                        )
                        if new_title != chart_config.get('title'):
                            st.session_state.dynamic_charts[i]['title'] = new_title
                    
                    with col2:
                        new_type = st.selectbox(
                            "Chart Type",
                            options=['line', 'scatter', 'bar', 'histogram', 'heatmap'],
                            index=['line', 'scatter', 'bar', 'histogram', 'heatmap'].index(chart_config.get('chart_type', 'line')),
                            key=f"type_{chart_config['id']}"
                        )
                        if new_type != chart_config.get('chart_type'):
                            st.session_state.dynamic_charts[i]['chart_type'] = new_type
                    
                    with col3:
                        if chart_config.get('chart_type', 'line') not in ['histogram', 'heatmap']:
                            x_options = ['timestamp'] + available_columns if 'timestamp' in df.columns else available_columns
                            current_x = chart_config.get('x_axis', x_options[0])
                            if current_x not in x_options and x_options:
                                current_x = x_options[0]
                            
                            if x_options:
                                new_x = st.selectbox(
                                    "X-Axis",
                                    options=x_options,
                                    index=x_options.index(current_x) if current_x in x_options else 0,
                                    key=f"x_{chart_config['id']}"
                                )
                                if new_x != chart_config.get('x_axis'):
                                    st.session_state.dynamic_charts[i]['x_axis'] = new_x
                    
                    with col4:
                        if chart_config.get('chart_type', 'line') != 'heatmap':
                            if available_columns:
                                current_y = chart_config.get('y_axis', available_columns[0])
                                if current_y not in available_columns:
                                    current_y = available_columns[0]
                                
                                new_y = st.selectbox(
                                    "Y-Axis",
                                    options=available_columns,
                                    index=available_columns.index(current_y) if current_y in available_columns else 0,
                                    key=f"y_{chart_config['id']}"
                                )
                                if new_y != chart_config.get('y_axis'):
                                    st.session_state.dynamic_charts[i]['y_axis'] = new_y
                    
                    with col5:
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete chart"):
                            try:
                                st.session_state.dynamic_charts.pop(i)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")
                    
                    # Display chart
                    try:
                        if chart_config.get('y_axis') or chart_config.get('chart_type') == 'heatmap':
                            fig = create_dynamic_chart(df, chart_config)
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config['id']}")
                        else:
                            st.warning("Please select a Y-axis variable for this chart.")
                    except Exception as e:
                        st.error(f"Error creating chart: {e}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error rendering chart {i}: {e}")

# Main optimized dashboard function
def main():
    """Optimized main dashboard function with better layout"""
    
    initialize_session_state()
    
    # Fixed header with connection status
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        status_class = "status-connected"
        status_text = "🟢 CONNECTED TO TELEMETRY STREAM"
    else:
        status_class = "status-disconnected" 
        status_text = "🔴 DISCONNECTED FROM TELEMETRY STREAM"
    
    st.markdown(f"""
    <div class="connection-banner {status_class}">
        {status_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Fixed header
    st.markdown("""
    <div class="fixed-header">
        <h1 class="main-title">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Compact connection management
    with st.sidebar:
        st.header("🔗 Connection")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Connect", use_container_width=True):
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
            if st.button("🛑 Stop", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    st.session_state.subscriber = None
                st.info("🛑 Disconnected")
                st.rerun()
        
        # Connection stats
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            stats = st.session_state.subscriber.get_stats()
            
            st.metric("📨 Messages", stats['messages_received'])
            st.metric("🔄 Attempts", stats['connection_attempts'])
            st.metric("❌ Errors", stats['errors'])
            
            if stats['last_message_time']:
                time_since = (datetime.now() - stats['last_message_time']).total_seconds()
                st.metric("⏰ Last Message", f"{time_since:.0f}s ago")
            
            if stats['last_error']:
                st.error(f"Last Error: {stats['last_error'][:40]}...")
        
        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.auto_refresh = st.checkbox(
            "🔄 Auto Refresh", 
            value=st.session_state.auto_refresh
        )
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Rate (s)", 1, 10, 3)
        
        st.info(f"📡 Channel: {CHANNEL_NAME}")
        
        # Data info
        if not st.session_state.telemetry_data.empty:
            st.subheader("📊 Data Info")
            st.metric("📋 Data Points", len(st.session_state.telemetry_data))
            st.metric("🕒 Last Update", st.session_state.last_update.strftime('%H:%M:%S'))
    
    # Get new messages and update data (optimized)
    new_messages_count = 0
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        new_messages = st.session_state.subscriber.get_messages()
        
        if new_messages:
            new_messages_count = len(new_messages)
            try:
                new_df = pd.DataFrame(new_messages)
                
                if 'timestamp' in new_df.columns:
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
                
                if st.session_state.telemetry_data.empty:
                    st.session_state.telemetry_data = new_df
                else:
                    st.session_state.telemetry_data = pd.concat([
                        st.session_state.telemetry_data, new_df
                    ], ignore_index=True)
                
                if len(st.session_state.telemetry_data) > MAX_DATAPOINTS:
                    st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_DATAPOINTS)
                
                st.session_state.last_update = datetime.now()
            except Exception as e:
                st.error(f"Error processing messages: {e}")
    
    # Main content
    df = st.session_state.telemetry_data.copy()
    
    if df.empty:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: var(--background-secondary); border-radius: 12px; margin: 2rem 0;">
            <h2>⏳ Waiting for telemetry data...</h2>
            <p style="font-size: 1.1rem; margin: 1rem 0;">
                1. Make sure <code>maindata.py</code> is running<br>
                2. Click 'Connect' to start receiving data
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 Debug Information"):
            st.json({
                "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                "Messages": st.session_state.subscriber.get_stats()['messages_received'] if st.session_state.subscriber else 0,
                "Channel": CHANNEL_NAME,
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set"
            })
    else:
        # KPIs at the top for better visibility
        try:
            kpis = calculate_kpis(df)
            render_kpi_dashboard(kpis)
        except Exception as e:
            st.error(f"Error calculating KPIs: {e}")
        
        if new_messages_count > 0:
            st.success(f"📨 +{new_messages_count} new messages received")
        
        # Charts in organized tabs
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🚀 Speed", "⚡ Power", "🎯 IMU-1", "📊 IMU-2", 
            "⚡ Efficiency", "🗺️ GPS", "📊 Custom", "📋 Data"
        ])
        
        with tab1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                st.plotly_chart(create_speed_chart(df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating speed chart: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                st.plotly_chart(create_power_chart(df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating power chart: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                st.plotly_chart(create_imu_chart(df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating IMU chart: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                st.plotly_chart(create_imu_chart_2(df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating detailed IMU chart: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                st.plotly_chart(create_efficiency_chart(df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating efficiency chart: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab6:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                st.plotly_chart(create_gps_map(df), use_container_width=True)
            except Exception as e:
                st.error(f"Error creating GPS map: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab7:
            try:
                render_dynamic_charts_section(df)
            except Exception as e:
                st.error(f"Error rendering dynamic charts: {e}")
        
        with tab8:
            st.subheader("📋 Raw Telemetry Data")
            try:
                st.dataframe(df.tail(100), use_container_width=True)
                
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error displaying data: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh logic (optimized)
    if st.session_state.auto_refresh and st.session_state.subscriber and st.session_state.subscriber.is_connected:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: var(--text-secondary); padding: 2rem;'>
        <p><strong>🏎️ Shell Eco-marathon Telemetry Dashboard</strong></p>
        <p>Real-time Data Visualization with Advanced IMU Integration & Custom Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
