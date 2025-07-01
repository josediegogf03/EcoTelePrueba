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

# Configuration
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATAPOINTS = 50000

# Page configuration - Optimized for better layout
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS with dual-theme support and improved layout
st.markdown("""
<style>
    /* Dual theme compatible colors */
    :root {
        --theme-bg-primary: #ffffff;
        --theme-bg-secondary: #f8f9fa;
        --theme-text-primary: #262730;
        --theme-text-secondary: #6c757d;
        --theme-accent: #ff6b35;
        --theme-success: #28a745;
        --theme-warning: #ffc107;
        --theme-danger: #dc3545;
        --theme-border: #dee2e6;
    }
    
    [data-theme="dark"] {
        --theme-bg-primary: #0e1117;
        --theme-bg-secondary: #262730;
        --theme-text-primary: #fafafa;
        --theme-text-secondary: #a3a8b8;
        --theme-accent: #ff6b35;
        --theme-success: #00d4aa;
        --theme-warning: #ffbd45;
        --theme-danger: #ff6b6b;
        --theme-border: #464853;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2rem;
        color: var(--theme-accent);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        padding: 1rem 0;
        border-bottom: 2px solid var(--theme-accent);
    }
    
    /* Status indicators */
    .status-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 2px solid;
        transition: all 0.3s ease;
    }
    
    .status-connected {
        background: linear-gradient(135deg, var(--theme-success) 0%, #20c997 100%);
        color: white;
        border-color: var(--theme-success);
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    
    .status-disconnected {
        background: linear-gradient(135deg, var(--theme-danger) 0%, #e74c3c 100%);
        color: white;
        border-color: var(--theme-danger);
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    
    .status-connecting {
        background: linear-gradient(135deg, var(--theme-warning) 0%, #f39c12 100%);
        color: #212529;
        border-color: var(--theme-warning);
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
    }
    
    /* Improved KPI cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .kpi-card {
        background: var(--theme-bg-secondary);
        border: 1px solid var(--theme-border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--theme-accent), #ff8c42);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--theme-text-primary);
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: var(--theme-text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    /* Chart containers - Fixed positioning */
    .chart-section {
        position: sticky;
        top: 0;
        z-index: 10;
        background: var(--theme-bg-primary);
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--theme-border);
    }
    
    .chart-container {
        background: var(--theme-bg-secondary);
        border: 1px solid var(--theme-border);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        min-height: 400px;
    }
    
    /* Dynamic chart builder */
    .dynamic-chart-builder {
        background: var(--theme-bg-secondary);
        border: 2px dashed var(--theme-accent);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .chart-config-row {
        display: flex;
        gap: 1rem;
        align-items: end;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .chart-config-row > div {
        flex: 1;
        min-width: 150px;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        .kpi-grid {
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
        }
        
        .kpi-card {
            padding: 1rem;
        }
        
        .kpi-value {
            font-size: 1.4rem;
        }
        
        .chart-config-row {
            flex-direction: column;
        }
        
        .chart-config-row > div {
            width: 100%;
        }
    }
    
    /* Animation and transitions */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--theme-bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--theme-text-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--theme-accent);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 20;
        background: var(--theme-bg-primary);
        padding: 0.5rem 0;
        border-bottom: 2px solid var(--theme-border);
        margin-bottom: 1rem;
    }
    
    /* Prevent scroll jump on updates */
    .main .block-container {
        scroll-behavior: smooth;
    }
    
    /* Performance optimizations */
    .chart-container canvas {
        will-change: transform;
    }
    
    /* Loading states */
    .loading-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        z-index: 100;
    }
    
    [data-theme="dark"] .loading-overlay {
        background: rgba(14, 17, 23, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Terminal logging setup (unchanged)
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

# TelemetrySubscriber class (keeping original implementation for reliability)
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
@st.cache_data
def get_initial_session_state():
    """Get initial session state configuration"""
    return {
        'telemetry_data': pd.DataFrame(),
        'last_update': datetime.now(),
        'auto_refresh': True,
        'dynamic_charts': [],
        'scroll_position': 0,
        'active_tab': 0,
        'theme_mode': 'auto'
    }

def initialize_session_state():
    """Initialize Streamlit session state with optimizations"""
    if 'subscriber' not in st.session_state:
        st.session_state.subscriber = None
    
    # Use cached initial state
    initial_state = get_initial_session_state()
    
    for key, value in initial_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Prevent scroll position reset on auto-refresh
    if 'prevent_scroll_reset' not in st.session_state:
        st.session_state.prevent_scroll_reset = False

# Optimized KPI calculations with caching
@st.cache_data(ttl=1)
def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators with caching"""
    if df.empty:
        return {
            'total_energy_mj': 0.0,
            'max_speed_ms': 0.0,
            'avg_speed_ms': 0.0,
            'total_distance_km': 0.0,
            'avg_power_w': 0.0,
            'efficiency_km_per_mj': 0.0,
            'max_acceleration': 0.0,
            'avg_gyro_magnitude': 0.0
        }
    
    # Vectorized calculations for better performance
    numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w', 'total_acceleration', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Use .iloc[-1] safely
    total_energy = df['energy_j'].iloc[-1] / 1_000_000 if 'energy_j' in df.columns and len(df) > 0 else 0
    total_distance = df['distance_m'].iloc[-1] / 1000 if 'distance_m' in df.columns and len(df) > 0 else 0
    
    # Vectorized operations
    max_speed = df['speed_ms'].max() if 'speed_ms' in df.columns else 0
    avg_speed = df['speed_ms'].mean() if 'speed_ms' in df.columns else 0
    avg_power = df['power_w'].mean() if 'power_w' in df.columns else 0
    efficiency = total_distance / total_energy if total_energy > 0 else 0
    
    max_acceleration = df['total_acceleration'].max() if 'total_acceleration' in df.columns else 0
    
    # Efficient gyroscope calculation
    if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
        gyro_magnitude = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        avg_gyro_magnitude = gyro_magnitude.mean()
    else:
        avg_gyro_magnitude = 0
    
    return {
        'total_energy_mj': total_energy,
        'max_speed_ms': max_speed,
        'avg_speed_ms': avg_speed,
        'total_distance_km': total_distance,
        'avg_power_w': avg_power,
        'efficiency_km_per_mj': efficiency,
        'max_acceleration': max_acceleration,
        'avg_gyro_magnitude': avg_gyro_magnitude
    }

# Enhanced KPI dashboard with better positioning
def render_kpi_dashboard(kpis: Dict[str, float]):
    """Render enhanced KPI dashboard with dual-theme support"""
    st.markdown("""
    <div class="kpi-grid fade-in">
        <div class="kpi-card">
            <div class="kpi-icon">🛣️</div>
            <div class="kpi-value">{:.2f}</div>
            <div class="kpi-label">Distance (km)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">⚡</div>
            <div class="kpi-value">{:.1f}</div>
            <div class="kpi-label">Max Speed (m/s)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🏃</div>
            <div class="kpi-value">{:.1f}</div>
            <div class="kpi-label">Avg Speed (m/s)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🔋</div>
            <div class="kpi-value">{:.2f}</div>
            <div class="kpi-label">Energy (MJ)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">💡</div>
            <div class="kpi-value">{:.1f}</div>
            <div class="kpi-label">Avg Power (W)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">♻️</div>
            <div class="kpi-value">{:.2f}</div>
            <div class="kpi-label">Efficiency (km/MJ)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value">{:.2f}</div>
            <div class="kpi-label">Max Accel (m/s²)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{:.2f}</div>
            <div class="kpi-label">Avg Gyro (°/s)</div>
        </div>
    </div>
    """.format(
        kpis['total_distance_km'],
        kpis['max_speed_ms'],
        kpis['avg_speed_ms'],
        kpis['total_energy_mj'],
        kpis['avg_power_w'],
        kpis['efficiency_km_per_mj'],
        kpis['max_acceleration'],
        kpis['avg_gyro_magnitude']
    ), unsafe_allow_html=True)

# Optimized chart creation with better performance
@st.cache_data(ttl=2)
def create_speed_chart(df: pd.DataFrame):
    """Create optimized speed chart"""
    if df.empty or 'speed_ms' not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    # Sample data for performance if too many points
    if len(df) > 1000:
        df_sampled = df.iloc[::max(1, len(df)//1000)]
    else:
        df_sampled = df
    
    fig = px.line(
        df_sampled, x='timestamp', y='speed_ms',
        title='🚗 Vehicle Speed Over Time',
        labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'},
        color_discrete_sequence=['#ff6b35']
    )
    fig.update_layout(
        height=350,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    fig.update_traces(line=dict(width=2))
    return fig

@st.cache_data(ttl=2)
def create_power_chart(df: pd.DataFrame):
    """Create optimized power system chart"""
    if df.empty or not all(col in df.columns for col in ['voltage_v', 'current_a', 'power_w']):
        return go.Figure().add_annotation(
            text="No power data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    # Sample data for performance
    if len(df) > 1000:
        df_sampled = df.iloc[::max(1, len(df)//1000)]
    else:
        df_sampled = df
    
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=('🔌 Voltage & Current', '⚡ Power'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(x=df_sampled['timestamp'], y=df_sampled['voltage_v'], 
                  name='Voltage (V)', line=dict(color='#3498db', width=2)), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sampled['timestamp'], y=df_sampled['current_a'], 
                  name='Current (A)', line=dict(color='#e74c3c', width=2)), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sampled['timestamp'], y=df_sampled['power_w'], 
                  name='Power (W)', line=dict(color='#2ecc71', width=2)), 
        row=2, col=1
    )
    
    fig.update_layout(
        height=450, 
        title_text="🔋 Electrical System Performance",
        template="plotly_white",
        margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig

@st.cache_data(ttl=2)
def create_imu_chart(df: pd.DataFrame):
    """Create optimized IMU chart"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    # Sample data for performance
    if len(df) > 1000:
        df_sampled = df.iloc[::max(1, len(df)//1000)]
    else:
        df_sampled = df
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🌀 Gyroscope Data (deg/s)', '📈 Accelerometer Data (m/s²)'),
        vertical_spacing=0.2
    )
    
    colors = {'x': '#e74c3c', 'y': '#2ecc71', 'z': '#3498db'}
    
    # Gyroscope data
    for axis, color in colors.items():
        fig.add_trace(
            go.Scatter(x=df_sampled['timestamp'], y=df_sampled[f'gyro_{axis}'], 
                      name=f'Gyro {axis.upper()}', line=dict(color=color, width=2)), 
            row=1, col=1
        )
    
    # Accelerometer data
    for axis, color in colors.items():
        fig.add_trace(
            go.Scatter(x=df_sampled['timestamp'], y=df_sampled[f'accel_{axis}'], 
                      name=f'Accel {axis.upper()}', line=dict(color=color, width=2)), 
            row=2, col=1
        )
      
    fig.update_layout(
        height=500, 
        title_text="🎯 IMU Sensor Data",
        template="plotly_white",
        margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig

@st.cache_data(ttl=2)
def create_efficiency_chart(df: pd.DataFrame):
    """Create optimized efficiency chart"""
    if df.empty or not all(col in df.columns for col in ['speed_ms', 'power_w']):
        return go.Figure().add_annotation(
            text="No efficiency data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    # Sample data for performance
    if len(df) > 500:
        df_sampled = df.iloc[::max(1, len(df)//500)]
    else:
        df_sampled = df
    
    fig = px.scatter(
        df_sampled, x='speed_ms', y='power_w',
        color='voltage_v' if 'voltage_v' in df_sampled.columns else None,
        title='♻️ Efficiency Analysis: Speed vs Power Consumption',
        labels={'speed_ms': 'Speed (m/s)', 'power_w': 'Power (W)'},
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        height=350,
        template="plotly_white",
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig

@st.cache_data(ttl=5)
def create_gps_map(df: pd.DataFrame):
    """Create optimized GPS map"""
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude']):
        return go.Figure().add_annotation(
            text="No GPS data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    if df_valid.empty:
        return go.Figure().add_annotation(
            text="No valid GPS coordinates",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    # Sample GPS data for performance
    if len(df_valid) > 200:
        df_valid = df_valid.iloc[::max(1, len(df_valid)//200)]
    
    center_point = dict(
        lat=df_valid['latitude'].mean(),
        lon=df_valid['longitude'].mean()
    )
    
    fig = px.scatter_map(
        df_valid, 
        lat='latitude', 
        lon='longitude',
        color='speed_ms' if 'speed_ms' in df_valid.columns else None,
        size='power_w' if 'power_w' in df_valid.columns else None,
        hover_data=['speed_ms', 'power_w', 'voltage_v'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w', 'voltage_v']) else None,
        map_style='carto-positron',
        title='🗺️ Vehicle Track and Performance',
        height=350,
        zoom=12,
        center=center_point,
        color_continuous_scale='viridis'
    )
    
    return fig

# Optimized dynamic charts with better performance
def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns available for plotting"""
    if df.empty:
        return []
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['message_id', 'uptime_seconds']
    return [col for col in numeric_columns if col not in exclude_cols]

@st.cache_data(ttl=2)
def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create optimized dynamic chart"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    x_col = chart_config.get('x_axis')
    y_col = chart_config.get('y_axis')
    chart_type = chart_config.get('chart_type', 'line')
    title = chart_config.get('title', f'{y_col} vs {x_col}')
    
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
    
    try:
        # Sample data for performance
        if len(df) > 1000:
            df_sampled = df.iloc[::max(1, len(df)//1000)]
        else:
            df_sampled = df
        
        if chart_type == 'line':
            fig = px.line(df_sampled, x=x_col, y=y_col, title=title,
                         color_discrete_sequence=['#ff6b35'])
        elif chart_type == 'scatter':
            fig = px.scatter(df_sampled, x=x_col, y=y_col, title=title,
                           color_discrete_sequence=['#ff6b35'])
        elif chart_type == 'bar':
            recent_df = df.tail(20)
            fig = px.bar(recent_df, x=x_col, y=y_col, title=title,
                        color_discrete_sequence=['#ff6b35'])
        elif chart_type == 'histogram':
            fig = px.histogram(df_sampled, x=y_col, title=f'Distribution of {y_col}',
                             color_discrete_sequence=['#ff6b35'])
        else:
            fig = px.line(df_sampled, x=x_col, y=y_col, title=title,
                         color_discrete_sequence=['#ff6b35'])
        
        fig.update_layout(
            height=350,
            template="plotly_white",
            margin=dict(t=50, b=40, l=40, r=40)
        )
        return fig
    
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )

# Enhanced dynamic charts section with better scroll handling
def render_dynamic_charts_section(df: pd.DataFrame):
    """Render enhanced dynamic charts section"""
    st.subheader("📊 Custom Chart Builder")
    
    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []
    
    if not available_columns:
        st.warning("No numeric data available for creating charts.")
        return
    
    # Improved layout for chart controls
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart",
                        use_container_width=True):
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
        
        with col2:
            if st.session_state.dynamic_charts:
                st.info(f"📈 {len(st.session_state.dynamic_charts)} custom chart(s) created")
    
    # Enhanced chart display with better layout
    if st.session_state.dynamic_charts:
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            try:
                with st.container():
                    st.markdown("---")
                    
                    # Chart configuration in a more compact layout
                    config_cols = st.columns([2, 1.5, 1.5, 1.5, 0.5])
                    
                    with config_cols[0]:
                        new_title = st.text_input(
                            "Chart Title", 
                            value=chart_config.get('title', 'New Chart'),
                            key=f"title_{chart_config['id']}"
                        )
                        if new_title != chart_config.get('title'):
                            st.session_state.dynamic_charts[i]['title'] = new_title
                    
                    with config_cols[1]:
                        new_type = st.selectbox(
                            "Chart Type",
                            options=['line', 'scatter', 'bar', 'histogram'],
                            index=['line', 'scatter', 'bar', 'histogram'].index(chart_config.get('chart_type', 'line')),
                            key=f"type_{chart_config['id']}"
                        )
                        if new_type != chart_config.get('chart_type'):
                            st.session_state.dynamic_charts[i]['chart_type'] = new_type
                    
                    with config_cols[2]:
                        if chart_config.get('chart_type', 'line') != 'histogram':
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
                    
                    with config_cols[3]:
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
                    
                    with config_cols[4]:
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete this chart"):
                            try:
                                st.session_state.dynamic_charts.pop(i)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")
                    
                    # Display the chart
                    try:
                        if chart_config.get('y_axis'):
                            with st.container():
                                fig = create_dynamic_chart(df, chart_config)
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config['id']}")
                        else:
                            st.warning("Please select a Y-axis variable for this chart.")
                    except Exception as e:
                        st.error(f"Error creating chart: {e}")
            
            except Exception as e:
                st.error(f"Error rendering chart {i}: {e}")
    
    else:
        st.markdown("""
        <div class="dynamic-chart-builder">
            <h4>🎯 Create Custom Charts</h4>
            <p>Click "Add Chart" to create custom visualizations with your preferred variables and chart types.</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div><strong>📈 Line:</strong> Time series data</div>  
                <div><strong>🔍 Scatter:</strong> Correlation analysis</div>
                <div><strong>📊 Bar:</strong> Recent value comparison</div>
                <div><strong>📈 Histogram:</strong> Data distribution</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main dashboard function with optimized layout
def main():
    """Main dashboard function with enhanced UI"""
    # Header with better styling
    st.markdown(
        '<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', 
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### 🔗 Connection Management")
        
        # Connection controls with better layout
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Connect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    time.sleep(2)
                
                with st.spinner("Connecting to Ably..."):
                    st.session_state.subscriber = TelemetrySubscriber()
                    if st.session_state.subscriber.connect():
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Connection failed!")
                
                st.rerun()
        
        with col2:
            if st.button("🛑 Disconnect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    st.session_state.subscriber = None
                st.info("🛑 Disconnected")
                st.rerun()
        
        # Enhanced connection status
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            st.markdown(
                '<div class="status-indicator status-connected">✅ Connected</div>',
                unsafe_allow_html=True
            )
            stats = st.session_state.subscriber.get_stats()
        else:
            st.markdown(
                '<div class="status-indicator status-disconnected">❌ Disconnected</div>',
                unsafe_allow_html=True
            )
            stats = {
                'messages_received': 0, 
                'connection_attempts': 0, 
                'errors': 0, 
                'last_message_time': None, 
                'last_error': None
            }
        
        # Enhanced stats display
        st.markdown("### 📊 Connection Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats['messages_received'])
            st.metric("Attempts", stats['connection_attempts'])
        with col2:
            st.metric("Errors", stats['errors'])
            if stats['last_message_time']:
                time_since = (datetime.now() - stats['last_message_time']).total_seconds()
                st.metric("Last Msg", f"{time_since:.0f}s ago")
            else:
                st.metric("Last Msg", "Never")
        
        if stats['last_error']:
            st.error(f"Last Error: {stats['last_error'][:50]}...")
        
        # Settings with better organization
        st.markdown("### ⚙️ Settings")
        new_auto_refresh = st.checkbox(
            "Auto Refresh", 
            value=st.session_state.auto_refresh
        )
        
        if new_auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = new_auto_refresh
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)
        
        st.info(f"📡 Channel: {CHANNEL_NAME}")
    
    # Main content processing
    new_messages_count = 0
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        new_messages = st.session_state.subscriber.get_messages()
        
        if new_messages:
            new_messages_count = len(new_messages)
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
    
    df = st.session_state.telemetry_data.copy()
    
    if df.empty:
        # Enhanced empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: var(--theme-bg-secondary); border-radius: 12px; margin: 2rem 0;">
            <h3>⏳ Waiting for telemetry data...</h3>
            <p style="margin: 1rem 0; color: var(--theme-text-secondary);">
                1. Make sure maindata.py is running<br>
                2. Click 'Connect' to start receiving data
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 Debug Info"):
            st.json({
                "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                "Messages": stats['messages_received'],
                "Errors": stats['errors'],
                "Channel": CHANNEL_NAME,
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set"
            })
    else:
        # KPI Dashboard at the top for immediate visibility
        kpis = calculate_kpis(df)
        render_kpi_dashboard(kpis)
        
        # Data info with better styling
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📊 {len(df)} data points | Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        with col2:
            if new_messages_count > 0:
                st.success(f"📨 +{new_messages_count} new")
        
        # Charts section with sticky navigation
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader("📈 Real-time Analytics")
        
        # Enhanced tabs with better organization
        tab_names = [
            "🚗 Speed", "🔋 Power", "🎯 IMU Data", 
            "♻️ Efficiency", "🗺️ GPS Track", 
            "📊 Custom Charts", "📋 Raw Data"
        ]
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:  # Speed Analysis
            with st.container():
                fig = create_speed_chart(df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:  # Power System
            with st.container():
                fig = create_power_chart(df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:  # IMU Sensors
            with st.container():
                fig = create_imu_chart(df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:  # Efficiency
            with st.container():
                fig = create_efficiency_chart(df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[4]:  # GPS Track
            with st.container():
                fig = create_gps_map(df)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[5]:  # Custom Charts
            render_dynamic_charts_section(df)
        
        with tabs[6]:  # Raw Data
            st.subheader("📋 Raw Telemetry Data")
            
            # Enhanced data display with filtering
            col1, col2 = st.columns([2, 1])
            with col1:
                show_rows = st.selectbox("Show rows", [50, 100, 200, 500], index=1)
            with col2:
                if st.button("📥 Download CSV", use_container_width=True):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download",
                        data=csv,
                        file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
            
            st.dataframe(df.tail(show_rows), use_container_width=True, height=400)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-refresh with better performance
    if st.session_state.auto_refresh and st.session_state.subscriber and st.session_state.subscriber.is_connected:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: var(--theme-text-secondary); padding: 1rem;'>
            <p><strong>Shell Eco-marathon Telemetry Dashboard</strong></p>
            <p>Real-time Data Visualization | Optimized for Performance</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
