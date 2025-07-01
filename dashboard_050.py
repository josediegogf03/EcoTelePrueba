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

# Optimized logging setup
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger('TelemetrySubscriber')
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
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
MAX_DATAPOINTS = 30000  # Reduced for better performance

# Page configuration
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optimized CSS for both light and dark themes with better UI
st.markdown("""
<style>
    :root {
        --primary-color: #00A693;
        --secondary-color: #6C7B7F;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        --info-color: #17a2b8;
        --light-bg: #ffffff;
        --dark-bg: #0e1117;
        --card-bg-light: #f8f9fa;
        --card-bg-dark: #262730;
        --text-primary-light: #212529;
        --text-primary-dark: #fafafa;
        --border-light: #dee2e6;
        --border-dark: #495057;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .connection-status {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .status-connected {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border-color: #c3e6cb;
        box-shadow: 0 2px 8px rgba(21, 87, 36, 0.15);
    }
    
    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border-color: #f5c6cb;
        box-shadow: 0 2px 8px rgba(114, 28, 36, 0.15);
    }
    
    .status-connecting {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border-color: #ffeaa7;
        box-shadow: 0 2px 8px rgba(133, 100, 4, 0.15);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .kpi-card {
        background: var(--card-bg-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
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
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: var(--secondary-color);
        margin-top: 5px;
        font-weight: 500;
    }
    
    .chart-container {
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        background: var(--card-bg-light);
        border: 1px solid var(--border-light);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 100;
        background: var(--light-bg);
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-light);
        backdrop-filter: blur(10px);
    }
    
    /* Dark theme adaptations */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background: var(--card-bg-dark);
            border-color: var(--border-dark);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .chart-container {
            background: var(--card-bg-dark);
            border-color: var(--border-dark);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        .sticky-header {
            background: var(--dark-bg);
            border-color: var(--border-dark);
        }
    }
    
    /* Reduced motion for accessibility */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* Performance optimizations */
    .plotly-graph-div {
        will-change: transform;
        contain: layout style paint;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        .kpi-card {
            margin: 0.25rem;
            padding: 1rem;
        }
        
        .kpi-value {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class TelemetrySubscriber:
    def __init__(self):
        self.ably_client = None
        self.channel = None
        self.is_connected = False
        self.message_queue = queue.Queue(maxsize=200)  # Limit queue size
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
        
        # Setup logging
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
            
            time.sleep(2)  # Reduced wait time
            
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
            
            with self._lock:
                # Optimized queue management
                try:
                    self.message_queue.put_nowait(data)
                except queue.Full:
                    # Remove oldest messages when queue is full
                    try:
                        for _ in range(50):  # Remove 50 oldest messages
                            self.message_queue.get_nowait()
                        self.message_queue.put_nowait(data)
                    except queue.Empty:
                        pass
                
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = datetime.now()
            
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
                self.connection_thread.join(timeout=3)  # Reduced timeout
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
        'refresh_interval': 3,
        'data_buffer': []  # Buffer for smoother updates
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data(ttl=60)  # Cache KPI calculations
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
    
    # Optimized calculations with error handling
    try:
        numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w', 'total_acceleration', 'gyro_x', 'gyro_y', 'gyro_z']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
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
            'total_energy_mj': round(total_energy, 3),
            'max_speed_ms': round(max_speed, 2),
            'avg_speed_ms': round(avg_speed, 2),
            'total_distance_km': round(total_distance, 3),
            'avg_power_w': round(avg_power, 1),
            'efficiency_km_per_mj': round(efficiency, 2),
            'max_acceleration': round(max_acceleration, 2),
            'avg_gyro_magnitude': round(avg_gyro_magnitude, 2)
        }
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return {key: 0.0 for key in ['total_energy_mj', 'max_speed_ms', 'avg_speed_ms', 'total_distance_km', 'avg_power_w', 'efficiency_km_per_mj', 'max_acceleration', 'avg_gyro_magnitude']}

def render_kpi_dashboard(kpis: Dict[str, float]):
    """Render optimized KPI dashboard"""
    # Sticky header for KPIs
    kpi_container = st.container()
    
    with kpi_container:
        st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
        st.subheader("📊 Performance Dashboard")
        
        # Optimized metrics with better organization
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🛣️ Distance",
                value=f"{kpis['total_distance_km']:.2f} km",
                help="Total distance traveled"
            )
            st.metric(
                label="📈 Max Accel",
                value=f"{kpis['max_acceleration']:.2f} m/s²",
                help="Maximum acceleration recorded"
            )
        
        with col2:
            st.metric(
                label="⚡ Max Speed",
                value=f"{kpis['max_speed_ms']:.1f} m/s",
                help="Maximum speed achieved"
            )
            st.metric(
                label="🎯 Avg Gyro",
                value=f"{kpis['avg_gyro_magnitude']:.2f} °/s",
                help="Average gyroscope magnitude"
            )
        
        with col3:
            st.metric(
                label="🏃 Avg Speed",
                value=f"{kpis['avg_speed_ms']:.1f} m/s",
                help="Average speed maintained"
            )
        
        with col4:
            st.metric(
                label="🔋 Energy",
                value=f"{kpis['total_energy_mj']:.2f} MJ",
                help="Total energy consumed"
            )
        
        # Secondary metrics row
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="💡 Avg Power",
                value=f"{kpis['avg_power_w']:.1f} W",
                help="Average power consumption"
            )
        
        with col2:
            st.metric(
                label="♻️ Efficiency",
                value=f"{kpis['efficiency_km_per_mj']:.2f} km/MJ",
                help="Energy efficiency ratio"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Optimized chart creation functions with better performance
@st.cache_data(ttl=30)
def create_speed_chart(df: pd.DataFrame):
    """Create optimized speed chart"""
    if df.empty or 'speed_ms' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No speed data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sample data for performance if too many points
    if len(df) > 1000:
        df_sampled = df.iloc[::len(df)//1000]
    else:
        df_sampled = df
    
    fig = px.line(
        df_sampled, x='timestamp', y='speed_ms',
        title='🚗 Vehicle Speed Over Time',
        labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(line=dict(width=2, color='#00A693'))
    
    return fig

@st.cache_data(ttl=30)
def create_power_chart(df: pd.DataFrame):
    """Create optimized power system chart"""
    if df.empty or not all(col in df.columns for col in ['voltage_v', 'current_a', 'power_w']):
        fig = go.Figure()
        fig.add_annotation(text="No power data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sample data for performance
    if len(df) > 1000:
        df_sampled = df.iloc[::len(df)//1000]
    else:
        df_sampled = df
    
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=('⚡ Voltage & Current', '💡 Power'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=df_sampled['timestamp'], y=df_sampled['voltage_v'], 
                  name='Voltage (V)', line=dict(color='#007bff', width=2)), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sampled['timestamp'], y=df_sampled['current_a'], 
                  name='Current (A)', line=dict(color='#dc3545', width=2)), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_sampled['timestamp'], y=df_sampled['power_w'], 
                  name='Power (W)', line=dict(color='#28a745', width=2)), 
        row=2, col=1
    )
    
    fig.update_layout(
        height=500, 
        title_text="🔋 Electrical System Performance",
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        template='plotly_white'
    )
    
    return fig

@st.cache_data(ttl=30)
def create_imu_chart(df: pd.DataFrame):
    """Create optimized IMU chart"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        fig = go.Figure()
        fig.add_annotation(text="No IMU data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sample data for performance
    if len(df) > 1000:
        df_sampled = df.iloc[::len(df)//1000]
    else:
        df_sampled = df
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🌀 Gyroscope Data (deg/s)', '📊 Accelerometer Data (m/s²)'),
        vertical_spacing=0.15
    )
    
    # Color scheme for better visibility
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Gyroscope data
    for i, (axis, color) in enumerate(zip(['gyro_x', 'gyro_y', 'gyro_z'], colors[:3])):
        fig.add_trace(
            go.Scatter(x=df_sampled['timestamp'], y=df_sampled[axis], 
                      name=f'Gyro {axis[-1].upper()}', line=dict(color=color, width=2)), 
            row=1, col=1
        )
    
    # Accelerometer data
    for i, (axis, color) in enumerate(zip(['accel_x', 'accel_y', 'accel_z'], colors[3:])):
        fig.add_trace(
            go.Scatter(x=df_sampled['timestamp'], y=df_sampled[axis], 
                      name=f'Accel {axis[-1].upper()}', line=dict(color=color, width=2)), 
            row=2, col=1
        )
    
    fig.update_layout(
        height=600, 
        title_text="🎯 IMU Sensor Data",
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        template='plotly_white'
    )
    
    return fig

@st.cache_data(ttl=30)
def create_efficiency_chart(df: pd.DataFrame):
    """Create optimized efficiency chart"""
    if df.empty or not all(col in df.columns for col in ['speed_ms', 'power_w']):
        fig = go.Figure()
        fig.add_annotation(text="No efficiency data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sample data and remove outliers for better visualization
    df_clean = df.dropna(subset=['speed_ms', 'power_w'])
    if len(df_clean) > 1000:
        df_clean = df_clean.sample(1000)
    
    fig = px.scatter(
        df_clean, x='speed_ms', y='power_w',
        color='voltage_v' if 'voltage_v' in df_clean.columns else None,
        title='⚡ Efficiency Analysis: Speed vs Power Consumption',
        labels={'speed_ms': 'Speed (m/s)', 'power_w': 'Power (W)'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

@st.cache_data(ttl=30)
def create_gps_map(df: pd.DataFrame):
    """Create optimized GPS map"""
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude']):
        fig = go.Figure()
        fig.add_annotation(text="No GPS data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid GPS coordinates",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Sample GPS points for performance
    if len(df_valid) > 500:
        df_valid = df_valid.iloc[::len(df_valid)//500]
    
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
        hover_data=['speed_ms', 'power_w'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w']) else None,
        map_style='open-street-map',
        title='🗺️ Vehicle Track and Performance',
        height=400,
        zoom=12,
        center=center_point
    )
    
    return fig

def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns available for plotting"""
    if df.empty:
        return []
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['message_id', 'uptime_seconds']
    return [col for col in numeric_columns if col not in exclude_cols]

@st.cache_data(ttl=30)
def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create optimized dynamic chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    x_col = chart_config.get('x_axis')
    y_col = chart_config.get('y_axis')
    chart_type = chart_config.get('chart_type', 'line')
    title = chart_config.get('title', f'{y_col} vs {x_col}')
    
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Invalid column selection",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    try:
        # Sample data for performance
        df_plot = df.dropna(subset=[x_col, y_col])
        if len(df_plot) > 1000:
            df_plot = df_plot.iloc[::len(df_plot)//1000]
        
        if chart_type == 'line':
            fig = px.line(df_plot, x=x_col, y=y_col, title=title, template='plotly_white')
        elif chart_type == 'scatter':
            fig = px.scatter(df_plot, x=x_col, y=y_col, title=title, template='plotly_white')
        elif chart_type == 'bar':
            recent_df = df_plot.tail(20)
            fig = px.bar(recent_df, x=x_col, y=y_col, title=title, template='plotly_white')
        elif chart_type == 'histogram':
            fig = px.histogram(df_plot, x=y_col, title=f'Distribution of {y_col}', template='plotly_white')
        else:
            fig = px.line(df_plot, x=x_col, y=y_col, title=title, template='plotly_white')
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {str(e)}",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

# Optimized dynamic charts with better state management
@st.fragment(run_every=5)
def render_dynamic_charts_section(df: pd.DataFrame):
    """Render dynamic charts section with auto-update"""
    st.subheader("📊 Dynamic Charts")
    
    try:
        available_columns = get_available_columns(df)
    except Exception as e:
        st.error(f"Error getting available columns: {e}")
        available_columns = []
    
    if not available_columns:
        st.warning("No numeric data available for creating charts.")
        return
    
    # Controls in columns for better layout
    col1, col2 = st.columns([1, 3])
    
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
    
    # Display charts with improved error handling
    for i, chart_config in enumerate(st.session_state.dynamic_charts):
        try:
            with st.container():
                st.markdown("---")
                
                # Chart configuration controls in a more compact layout
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 0.5])
                
                with col1:
                    new_title = st.text_input(
                        "Chart Title", 
                        value=chart_config.get('title', 'New Chart'),
                        key=f"title_{chart_config['id']}"
                    )
                    if new_title != chart_config.get('title'):
                        st.session_state.dynamic_charts[i]['title'] = new_title
                
                with col2:
                    chart_types = ['line', 'scatter', 'bar', 'histogram']
                    current_type = chart_config.get('chart_type', 'line')
                    type_index = chart_types.index(current_type) if current_type in chart_types else 0
                    
                    new_type = st.selectbox(
                        "Type",
                        options=chart_types,
                        index=type_index,
                        key=f"type_{chart_config['id']}"
                    )
                    if new_type != chart_config.get('chart_type'):
                        st.session_state.dynamic_charts[i]['chart_type'] = new_type
                
                with col3:
                    if chart_config.get('chart_type', 'line') != 'histogram':
                        x_options = ['timestamp'] + available_columns if 'timestamp' in df.columns else available_columns
                        current_x = chart_config.get('x_axis', x_options[0] if x_options else '')
                        if current_x not in x_options and x_options:
                            current_x = x_options[0]
                        
                        if x_options:
                            x_index = x_options.index(current_x) if current_x in x_options else 0
                            new_x = st.selectbox(
                                "X-Axis",
                                options=x_options,
                                index=x_index,
                                key=f"x_{chart_config['id']}"
                            )
                            if new_x != chart_config.get('x_axis'):
                                st.session_state.dynamic_charts[i]['x_axis'] = new_x
                
                with col4:
                    if available_columns:
                        current_y = chart_config.get('y_axis', available_columns[0])
                        if current_y not in available_columns:
                            current_y = available_columns[0]
                        
                        y_index = available_columns.index(current_y) if current_y in available_columns else 0
                        new_y = st.selectbox(
                            "Y-Axis",
                            options=available_columns,
                            index=y_index,
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
                
                # Display the chart
                try:
                    if chart_config.get('y_axis'):
                        fig = create_dynamic_chart(df, chart_config)
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_config['id']}")
                    else:
                        st.warning("Please select a Y-axis variable for this chart.")
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
        
        except Exception as e:
            st.error(f"Error rendering chart {i}: {e}")

def main():
    """Optimized main dashboard function"""
    st.markdown(
        '<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', 
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    
    # Sidebar with improved layout
    with st.sidebar:
        st.header("🔗 Connection Management")
        
        # Connection controls in a more compact layout
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Connect", use_container_width=True):
                if st.session_state.subscriber:
                    st.session_state.subscriber.disconnect()
                    time.sleep(1)
                
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
        
        # Connection status with better styling
        if st.session_state.subscriber and st.session_state.subscriber.is_connected:
            st.markdown(
                '<div class="connection-status status-connected">✅ Connected</div>',
                unsafe_allow_html=True
            )
            stats = st.session_state.subscriber.get_stats()
        else:
            st.markdown(
                '<div class="connection-status status-disconnected">❌ Disconnected</div>',
                unsafe_allow_html=True
            )
            stats = {
                'messages_received': 0, 
                'connection_attempts': 0, 
                'errors': 0, 
                'last_message_time': None, 
                'last_error': None
            }
        
        # Improved stats display
        with st.expander("📊 Connection Stats", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", stats['messages_received'])
                st.metric("Attempts", stats['connection_attempts'])
            with col2:
                st.metric("Errors", stats['errors'])
                if stats['last_message_time']:
                    time_since = (datetime.now() - stats['last_message_time']).total_seconds()
                    st.metric("Last Msg", f"{time_since:.0f}s")
                else:
                    st.metric("Last Msg", "Never")
        
        if stats['last_error']:
            st.error(f"Last Error: {stats['last_error'][:50]}...")
        
        # Settings with better organization
        with st.expander("⚙️ Settings", expanded=True):
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh", 
                value=st.session_state.auto_refresh
            )
            
            if st.session_state.auto_refresh:
                st.session_state.refresh_interval = st.slider(
                    "Refresh Interval (s)", 
                    1, 10, 
                    st.session_state.refresh_interval
                )
        
        st.info(f"📡 Channel: {CHANNEL_NAME}")
    
    # Optimized data processing
    new_messages_count = 0
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        new_messages = st.session_state.subscriber.get_messages()
        
        if new_messages:
            new_messages_count = len(new_messages)
            new_df = pd.DataFrame(new_messages)
            
            # Optimized timestamp processing
            if 'timestamp' in new_df.columns:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')
            
            # Efficient data concatenation
            if st.session_state.telemetry_data.empty:
                st.session_state.telemetry_data = new_df
            else:
                st.session_state.telemetry_data = pd.concat([
                    st.session_state.telemetry_data, new_df
                ], ignore_index=True)
            
            # Efficient data trimming
            if len(st.session_state.telemetry_data) > MAX_DATAPOINTS:
                st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_DATAPOINTS)
            
            st.session_state.last_update = datetime.now()
    
    # Main content with better organization
    df = st.session_state.telemetry_data.copy()
    
    if df.empty:
        # Improved empty state
        st.warning("⏳ Waiting for telemetry data...")
        
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                📋 **Setup Instructions:**
                1. Make sure maindata.py is running
                2. Click 'Connect' to start receiving data
                3. Wait for telemetry messages to arrive
                """)
            
            with col2:
                with st.expander("🔍 Debug Info"):
                    st.json({
                        "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                        "Messages": stats['messages_received'],
                        "Errors": stats['errors'],
                        "Channel": CHANNEL_NAME,
                        "Data Points": len(df)
                    })
    else:
        # Display success message for new data
        if new_messages_count > 0:
            st.success(f"📨 Received {new_messages_count} new messages | 📊 {len(df)} total data points")
        
        # KPI Dashboard
        kpis = calculate_kpis(df)
        render_kpi_dashboard(kpis)
        
        # Status info
        st.info(f"📊 {len(df)} data points | Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Optimized tabs with better organization
        st.subheader("📈 Real-time Analytics")
        
        tabs = st.tabs([
            "🚗 Speed Analysis", 
            "🔋 Power System", 
            "🎯 IMU Sensors", 
            "⚡ Efficiency", 
            "🗺️ GPS Track", 
            "📊 Dynamic Charts", 
            "📄 Raw Data"
        ])
        
        with tabs[0]:
            with st.container():
                st.plotly_chart(create_speed_chart(df), use_container_width=True)
        
        with tabs[1]:
            with st.container():
                st.plotly_chart(create_power_chart(df), use_container_width=True)
        
        with tabs[2]:
            with st.container():
                st.plotly_chart(create_imu_chart(df), use_container_width=True)
        
        with tabs[3]:
            with st.container():
                st.plotly_chart(create_efficiency_chart(df), use_container_width=True)
        
        with tabs[4]:
            with st.container():
                st.plotly_chart(create_gps_map(df), use_container_width=True)
        
        with tabs[5]:
            with st.container():
                render_dynamic_charts_section(df)
        
        with tabs[6]:
            with st.container():
                st.subheader("📄 Raw Telemetry Data")
                
                # Display controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    show_rows = st.selectbox("Show rows", [50, 100, 200, 500], index=1)
                
                with col2:
                    if st.button("🔄 Refresh Data"):
                        st.rerun()
                
                with col3:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Optimized data display
                st.dataframe(
                    df.tail(show_rows), 
                    use_container_width=True,
                    height=400
                )
    
    # Auto-refresh with optimized timing
    if (st.session_state.auto_refresh and 
        st.session_state.subscriber and 
        st.session_state.subscriber.is_connected):
        time.sleep(st.session_state.refresh_interval)
        st.rerun()
    
    # Improved footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6C7B7F; padding: 1rem;'>
        <p><strong>Shell Eco-marathon Telemetry Dashboard</strong> | 
        Real-time Data Visualization with Advanced Analytics</p>
        <p>🚗 Optimized for Performance | 🎯 Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
