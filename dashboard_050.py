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

# Setup terminal logging
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

# Page configuration
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS for better theming and scroll behavior
st.markdown("""
<style>
    /* Main styling for both themes */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color, #4CAF50);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        position: sticky;
        top: 0;
        background: var(--background-color);
        z-index: 1000;
        padding: 1rem 0;
        border-bottom: 2px solid var(--secondary-background-color);
    }
    
    /* Connection status with theme-aware colors */
    .connection-status {
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s ease;
    }
    .status-connected {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border: 1px solid #1e7e34;
    }
    .status-disconnected {
        background: linear-gradient(135deg, #dc3545, #e74c3c);
        color: white;
        border: 1px solid #c82333;
    }
    .status-connecting {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: #212529;
        border: 1px solid #e0a800;
    }
    
    /* KPI cards with enhanced styling */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
        position: sticky;
        top: 120px;
        z-index: 999;
        padding: 1rem;
        background: var(--background-color);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .kpi-card {
        background: linear-gradient(135deg, var(--secondary-background-color), var(--background-color));
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color, #4CAF50);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color, #4CAF50), transparent);
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--text-color);
        margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Chart container with improved scroll behavior */
    .chart-container {
        position: relative;
        margin: 1rem 0;
        scroll-margin-top: 200px;
    }
    
    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 200px;
        z-index: 998;
        background: var(--background-color);
        border-bottom: 2px solid var(--secondary-background-color);
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    /* Dynamic charts styling */
    .dynamic-chart-container {
        border: 2px dashed var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, var(--secondary-background-color), var(--background-color));
        position: relative;
        overflow: hidden;
    }
    
    .dynamic-chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color, #4CAF50), var(--secondary-background-color));
    }
    
    /* Scroll behavior fixes */
    .main .block-container {
        scroll-behavior: smooth;
        padding-top: 0;
    }
    
    /* Prevent layout shifts */
    .element-container {
        scroll-margin-top: 220px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-online {
        background-color: #28a745;
    }
    
    .status-offline {
        background-color: #dc3545;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            top: 100px;
        }
        
        .main-header {
            font-size: 1.8rem;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            top: 150px;
        }
    }
    
    /* Performance optimizations */
    .plotly-chart {
        will-change: transform;
    }
    
    /* Loading states */
    .loading-spinner {
        border: 4px solid var(--secondary-background-color);
        border-top: 4px solid var(--primary-color, #4CAF50);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

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

def initialize_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        'subscriber': None,
        'telemetry_data': pd.DataFrame(),
        'last_update': datetime.now(),
        'auto_refresh': True,
        'dynamic_charts': [],
        'active_tab': 0,
        'scroll_position': 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@st.cache_data(ttl=5)
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
    
    # Optimize numeric conversion
    numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w', 'total_acceleration', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Use vectorized operations for better performance
    total_energy = df['energy_j'].iloc[-1] / 1_000_000 if 'energy_j' in df.columns and len(df) > 0 else 0
    max_speed = df['speed_ms'].max() if 'speed_ms' in df.columns else 0
    avg_speed = df['speed_ms'].mean() if 'speed_ms' in df.columns else 0
    total_distance = df['distance_m'].iloc[-1] / 1000 if 'distance_m' in df.columns and len(df) > 0 else 0
    avg_power = df['power_w'].mean() if 'power_w' in df.columns else 0
    efficiency = total_distance / total_energy if total_energy > 0 else 0
    
    max_acceleration = df['total_acceleration'].max() if 'total_acceleration' in df.columns else 0
    
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

def render_kpi_dashboard(kpis: Dict[str, float]):
    """Render sticky KPI dashboard"""
    kpi_data = [
        ("🛣️", "Distance", f"{kpis['total_distance_km']:.2f} km"),
        ("⚡", "Max Speed", f"{kpis['max_speed_ms']:.1f} m/s"),
        ("🏃", "Avg Speed", f"{kpis['avg_speed_ms']:.1f} m/s"),
        ("🔋", "Energy", f"{kpis['total_energy_mj']:.2f} MJ"),
        ("💡", "Avg Power", f"{kpis['avg_power_w']:.1f} W"),
        ("♻️", "Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ"),
        ("📈", "Max Accel", f"{kpis['max_acceleration']:.2f} m/s²"),
        ("🎯", "Avg Gyro", f"{kpis['avg_gyro_magnitude']:.2f} °/s")
    ]
    
    # Create HTML grid for KPIs
    kpi_html = '<div class="kpi-grid">'
    for icon, label, value in kpi_data:
        kpi_html += f'''
        <div class="kpi-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                <span class="kpi-label">{label}</span>
            </div>
            <div class="kpi-value">{value}</div>
        </div>
        '''
    kpi_html += '</div>'
    
    st.markdown(kpi_html, unsafe_allow_html=True)

@st.cache_data(ttl=10)
def create_speed_chart(df_hash: str, df: pd.DataFrame):
    """Create cached speed chart"""
    if df.empty or 'speed_ms' not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.line(
        df, x='timestamp', y='speed_ms',
        title='🚗 Vehicle Speed Over Time',
        labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'},
        color_discrete_sequence=['#4CAF50']
    )
    fig.update_layout(
        height=400,
        template='plotly_white',
        hovermode='x unified',
        showlegend=False
    )
    return fig

@st.cache_data(ttl=10)
def create_power_chart(df_hash: str, df: pd.DataFrame):
    """Create cached power chart"""
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
                  name='Voltage (V)', line=dict(color='#2196F3')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['current_a'], 
                  name='Current (A)', line=dict(color='#FF5722')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_w'], 
                  name='Power (W)', line=dict(color='#4CAF50')), 
        row=2, col=1
    )
    
    fig.update_layout(
        height=500, 
        title_text="⚡ Electrical System Performance",
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

@st.cache_data(ttl=10)
def create_imu_chart(df_hash: str, df: pd.DataFrame):
    """Create cached IMU chart"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🌪️ Gyroscope Data (deg/s)', '📐 Accelerometer Data (m/s²)'),
        vertical_spacing=0.25
    )
    
    colors = {'x': '#FF5722', 'y': '#4CAF50', 'z': '#2196F3'}
    
    for axis, color in colors.items():
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'gyro_{axis}'], 
                      name=f'Gyro {axis.upper()}', line=dict(color=color)), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'accel_{axis}'], 
                      name=f'Accel {axis.upper()}', line=dict(color=color, dash='dash')), 
            row=2, col=1
        )
      
    fig.update_layout(
        height=600, 
        title_text="🎯 IMU Sensor Data",
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

@st.cache_data(ttl=10)
def create_imu_chart_2(df_hash: str, df: pd.DataFrame):
    """Create cached detailed IMU chart"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z'),
        vertical_spacing=0.3,
        horizontal_spacing=0.1
    )

    colors = ['#FF5722', '#4CAF50', '#2196F3']
    axes = ['x', 'y', 'z']
    
    for i, (axis, color) in enumerate(zip(axes, colors)):
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'gyro_{axis}'], 
                      name=f'Gyro {axis.upper()}', line=dict(color=color),
                      showlegend=False), 
            row=1, col=i+1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df[f'accel_{axis}'], 
                      name=f'Accel {axis.upper()}', line=dict(color=color),
                      showlegend=False), 
            row=2, col=i+1
        )

    fig.update_layout(
        height=600, 
        title_text="📊 Detailed IMU Analysis",
        template='plotly_white'
    )
    return fig

@st.cache_data(ttl=10)
def create_efficiency_chart(df_hash: str, df: pd.DataFrame):
    """Create cached efficiency chart"""
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
        template='plotly_white',
        hovermode='closest'
    )
    return fig

@st.cache_data(ttl=10)
def create_gps_map(df_hash: str, df: pd.DataFrame):
    """Create cached GPS map"""
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
    
    fig = px.scatter_map(
        df_valid, 
        lat='latitude', 
        lon='longitude',
        color='speed_ms' if 'speed_ms' in df_valid.columns else None,
        size='power_w' if 'power_w' in df_valid.columns else None,
        hover_data=['speed_ms', 'power_w', 'voltage_v'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w', 'voltage_v']) else None,
        map_style='open-street-map',
        title='🗺️ Vehicle Track and Performance',
        height=400,
        zoom=12,
        center=center_point,
        color_continuous_scale='Viridis'
    )
    
    return fig

@st.cache_data(ttl=5)
def get_available_columns(df_hash: str, df: pd.DataFrame) -> List[str]:
    """Get cached list of available columns"""
    if df.empty:
        return []
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['message_id', 'uptime_seconds']
    return [col for col in numeric_columns if col not in exclude_cols]

def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create dynamic chart with improved error handling"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
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
    
    try:
        color_sequence = ['#4CAF50', '#2196F3', '#FF5722', '#FF9800', '#9C27B0']
        
        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=color_sequence)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=color_sequence)
        elif chart_type == 'bar':
            recent_df = df.tail(20)
            fig = px.bar(recent_df, x=x_col, y=y_col, title=title, color_discrete_sequence=color_sequence)
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=y_col, title=f'Distribution of {y_col}', color_discrete_sequence=color_sequence)
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, color_discrete_sequence=color_sequence)
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            hovermode='closest'
        )
        return fig
    
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@st.fragment(run_every=3)
def render_dynamic_charts_section(df: pd.DataFrame):
    """Optimized dynamic charts section with caching"""
    st.subheader("📊 Dynamic Charts")
    
    # Cache DataFrame hash for performance
    df_hash = str(hash(df.to_string())) if not df.empty else "empty"
    available_columns = get_available_columns(df_hash, df) if not df.empty else []
    
    if not available_columns:
        st.warning("No numeric data available for creating charts.")
        return
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart"):
            try:
                new_chart = {
                    'id': str(uuid.uuid4()),
                    'title': 'New Chart',
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
    
    # Display charts
    if st.session_state.dynamic_charts:
        for i, chart_config in enumerate(st.session_state.dynamic_charts):
            try:
                with st.container(border=True, key=f"chart_container_{chart_config['id']}"):
                    # Chart controls
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    
                    with col1:
                        new_title = st.text_input(
                            "Chart Title", 
                            value=chart_config.get('title', 'New Chart'),
                            key=f"title_{chart_config['id']}"
                        )
                        if new_title != chart_config.get('title'):
                            st.session_state.dynamic_charts[i]['title'] = new_title
                    
                    with col2:
                        new_type = st.selectbox(
                            "Chart Type",
                            options=['line', 'scatter', 'bar', 'histogram'],
                            index=['line', 'scatter', 'bar', 'histogram'].index(chart_config.get('chart_type', 'line')),
                            key=f"type_{chart_config['id']}"
                        )
                        if new_type != chart_config.get('chart_type'):
                            st.session_state.dynamic_charts[i]['chart_type'] = new_type
                    
                    with col3:
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
                    
                    with col4:
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
                        if st.button("🗑️", key=f"delete_{chart_config['id']}", help="Delete this chart"):
                            try:
                                st.session_state.dynamic_charts.pop(i)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting chart: {e}")
                    
                    # Display chart
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
        <div class="dynamic-chart-container">
            <h4>🎯 Create Custom Charts</h4>
            <p>Click "Add Chart" to create custom visualizations with your preferred variables and chart types.</p>
            <p><strong>Available chart types:</strong></p>
            <ul>
                <li><strong>Line:</strong> Great for time series data</li>
                <li><strong>Scatter:</strong> Perfect for correlation analysis</li>
                <li><strong>Bar:</strong> Good for comparing recent values</li>
                <li><strong>Histogram:</strong> Shows data distribution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Optimized main dashboard function"""
    # Header with status indicator
    connection_status = "🟢" if (st.session_state.get('subscriber') and st.session_state.subscriber.is_connected) else "🔴"
    st.markdown(
        f'<h1 class="main-header">{connection_status} Shell Eco-marathon Telemetry Dashboard</h1>', 
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    
    # Sidebar with improved layout
    with st.sidebar:
        st.header("🔗 Connection Management")
        
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
        
        # Connection status
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            st.markdown(
                '<div class="connection-status status-connected"><span class="status-indicator status-online"></span>Connected</div>',
                unsafe_allow_html=True
            )
            stats = st.session_state.subscriber.get_stats()
        else:
            st.markdown(
                '<div class="connection-status status-disconnected"><span class="status-indicator status-offline"></span>Disconnected</div>',
                unsafe_allow_html=True
            )
            stats = {
                'messages_received': 0, 
                'connection_attempts': 0, 
                'errors': 0, 
                'last_message_time': None, 
                'last_error': None
            }
        
        # Stats display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📨 Messages", stats['messages_received'])
            st.metric("🔄 Attempts", stats['connection_attempts'])
        with col2:
            st.metric("❌ Errors", stats['errors'])
            if stats['last_message_time']:
                time_since = (datetime.now() - stats['last_message_time']).total_seconds()
                st.metric("⏱️ Last Msg", f"{time_since:.0f}s ago")
            else:
                st.metric("⏱️ Last Msg", "Never")
        
        if stats['last_error']:
            st.error(f"Last Error: {stats['last_error'][:50]}...")
        
        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.auto_refresh = st.checkbox(
            "Auto Refresh", 
            value=st.session_state.auto_refresh
        )
        
        if st.session_state.auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 10, 3)
        
        st.info(f"📡 Channel: {CHANNEL_NAME}")
    
    # Process new messages
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
    
    # Main content
    df = st.session_state.telemetry_data.copy()
    
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        col1, col2 = st.columns(2)
        with col1:
            st.info("1. Make sure maindata.py is running")
        with col2:
            st.info("2. Click 'Connect' to start receiving data")
        
        with st.expander("🔍 Debug Info"):
            st.json({
                "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                "Messages": stats['messages_received'],
                "Errors": stats['errors'],
                "Channel": CHANNEL_NAME,
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set"
            })
    else:
        # Calculate and display KPIs (now sticky at top)
        kpis = calculate_kpis(df)
        render_kpi_dashboard(kpis)
        
        # Status info
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"📊 {len(df)} data points")
        with col2:
            st.info(f"🕒 Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        with col3:
            if new_messages_count > 0:
                st.success(f"📨 +{new_messages_count}")
        
        # Charts with improved caching
        st.subheader("📈 Real-time Analytics")
        
        df_hash = str(hash(df.to_string())) if not df.empty else "empty"
        
        tab_names = ["Speed Analysis", "Power System", "IMU Sensors", "IMU Details", "Efficiency", "GPS Track", "Dynamic Charts", "Raw Data"]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            with st.container():
                st.plotly_chart(create_speed_chart(df_hash, df), use_container_width=True)
        
        with tabs[1]:
            with st.container():
                st.plotly_chart(create_power_chart(df_hash, df), use_container_width=True)
        
        with tabs[2]:
            with st.container():
                st.plotly_chart(create_imu_chart(df_hash, df), use_container_width=True)

        with tabs[3]:
            with st.container():
                st.plotly_chart(create_imu_chart_2(df_hash, df), use_container_width=True)
        
        with tabs[4]:
            with st.container():
                st.plotly_chart(create_efficiency_chart(df_hash, df), use_container_width=True)
        
        with tabs[5]:
            with st.container():
                st.plotly_chart(create_gps_map(df_hash, df), use_container_width=True)
        
        with tabs[6]:
            render_dynamic_charts_section(df)
        
        with tabs[7]:
            st.subheader("📋 Raw Telemetry Data")
            st.dataframe(df.tail(100), use_container_width=True, height=400)
            
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
    
    # Auto-refresh
    if st.session_state.auto_refresh and st.session_state.subscriber and st.session_state.subscriber.is_connected:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: var(--text-color); opacity: 0.7;'>"
        "<p>🏎️ Shell Eco-marathon Telemetry Dashboard | Real-time Data Visualization</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
