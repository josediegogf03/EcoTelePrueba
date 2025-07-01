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

# --- UPDATED: Function to set up terminal logging ---
def setup_terminal_logging():
    """Configures the 'TelemetrySubscriber' logger to print to the terminal."""
    logger = logging.getLogger('TelemetrySubscriber')
    
    # Prevent adding handlers multiple times on Streamlit reruns
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# --- Call the logging setup function once at the start ---
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

# Custom CSS - Updated to prevent scrolling issues and maintain scroll position
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .connection-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
        text-align: center;
    }
    .status-connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-connecting {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .dynamic-chart-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .add-chart-button {
        font-size: 1.2rem;
        padding: 10px 20px;
        margin: 10px;
    }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 5px;
    }
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    /* SCROLL POSITION FIX: Prevent automatic scroll to top on rerun */
    .main .block-container {
        scroll-behavior: smooth;
    }
    
    /* Maintain focus and scroll position */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 0;
        z-index: 10;
        background-color: white;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Prevent unwanted scroll jumps */
    .element-container {
        scroll-margin-top: 100px;
    }
    
    @media (max-width: 768px) {
        .kpi-container {
            flex-direction: column;
        }
    }
</style>

<script>
    // JavaScript to maintain scroll position on rerun
    function preserveScrollPosition() {
        // Store scroll position before rerun
        sessionStorage.setItem('scrollY', window.scrollY.toString());
        sessionStorage.setItem('scrollX', window.scrollX.toString());
    }
    
    function restoreScrollPosition() {
        // Restore scroll position after rerun
        const scrollY = sessionStorage.getItem('scrollY');
        const scrollX = sessionStorage.getItem('scrollX');
        if (scrollY && scrollX) {
            setTimeout(() => {
                window.scrollTo(parseInt(scrollX), parseInt(scrollY));
            }, 100);
        }
    }
    
    // Listen for beforeunload to save scroll position
    window.addEventListener('beforeunload', preserveScrollPosition);
    
    // Restore scroll position when page loads
    document.addEventListener('DOMContentLoaded', restoreScrollPosition);
    
    // Also try to restore on Streamlit rerun
    setTimeout(restoreScrollPosition, 500);
</script>
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
        
        # Setup logging
        self.logger = logging.getLogger('TelemetrySubscriber')
    
    def connect(self) -> bool:
        """Connect to Ably and start receiving messages"""
        try:
            with self._lock:
                self.stats['connection_attempts'] += 1
            
            self.logger.info("🔄 Starting connection to Ably...")
            
            # Stop any existing connection
            if self._should_run:
                self.disconnect()
            
            # Clear stop event and start
            self._stop_event.clear()
            self._should_run = True
            
            # Start connection thread
            self.connection_thread = threading.Thread(target=self._connection_worker, daemon=True)
            self.connection_thread.start()
            
            # Wait a bit for connection to establish
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
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the connection coroutine
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
            
            # Create Ably client
            self.ably_client = AblyRealtime(ABLY_API_KEY)
            
            # Setup connection event handlers
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
            
            # Attach connection event handlers
            self.ably_client.connection.on('connected', on_connected)
            self.ably_client.connection.on('disconnected', on_disconnected)
            self.ably_client.connection.on('failed', on_failed)
            self.ably_client.connection.on('suspended', on_disconnected)
            
            # Wait for connection to establish
            self.logger.info("⏳ Waiting for connection...")
            await self.ably_client.connection.once_async('connected')
            
            # Get channel and subscribe
            self.logger.info(f"📡 Getting channel: {CHANNEL_NAME}")
            self.channel = self.ably_client.channels.get(CHANNEL_NAME)
            
            # Subscribe to messages
            self.logger.info("📨 Subscribing to messages...")
            await self.channel.subscribe('telemetry_update', self._on_message_received)
            
            self.logger.info("✅ Successfully subscribed to messages!")
            
            # Keep connection alive
            while self._should_run and not self._stop_event.is_set():
                await asyncio.sleep(1)
                
                # Check connection state
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
            
            # Extract message data
            data = message.data
            
            # Parse JSON if it's a string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON decode error: {e}")
                    with self._lock:
                        self.stats['errors'] += 1
                        self.stats['last_error'] = f"JSON decode error: {e}"
                    return
            
            # Validate data
            if not isinstance(data, dict):
                self.logger.warning(f"⚠️ Invalid data type: {type(data)}")
                with self._lock:
                    self.stats['errors'] += 1
                    self.stats['last_error'] = f"Invalid data type: {type(data)}"
                return
            
            self.logger.debug(f"📊 Data keys: {list(data.keys())}")
            
            # Add to message queue
            with self._lock:
                # Prevent queue from growing too large
                if self.message_queue.qsize() > 100:
                    try:
                        # Remove old messages
                        while self.message_queue.qsize() > 50:
                            self.message_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add new message
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
            
            # Stop the connection loop
            self._should_run = False
            self._stop_event.set()
            self.is_connected = False
            
            # Close Ably connection
            if self.ably_client:
                try:
                    self.ably_client.close()
                    self.logger.info("✅ Ably connection closed")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error closing Ably: {e}")
            
            # Wait for thread to finish
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
    if 'subscriber' not in st.session_state:
        st.session_state.subscriber = None
    
    if 'telemetry_data' not in st.session_state:
        st.session_state.telemetry_data = pd.DataFrame()
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'dynamic_charts' not in st.session_state:
        st.session_state.dynamic_charts = []
    
    # Add session state for scroll position management
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Track if this is from an auto-refresh to prevent scroll jump
    if 'is_auto_refresh' not in st.session_state:
        st.session_state.is_auto_refresh = False

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators"""
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
    
    # Ensure numeric columns
    numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w', 'total_acceleration', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    total_energy = df['energy_j'].iloc[-1] / 1_000_000 if 'energy_j' in df.columns and len(df) > 0 else 0
    max_speed = df['speed_ms'].max() if 'speed_ms' in df.columns else 0
    avg_speed = df['speed_ms'].mean() if 'speed_ms' in df.columns else 0
    total_distance = df['distance_m'].iloc[-1] / 1000 if 'distance_m' in df.columns and len(df) > 0 else 0
    avg_power = df['power_w'].mean() if 'power_w' in df.columns else 0
    efficiency = total_distance / total_energy if total_energy > 0 else 0
    
    # IMU-based KPIs
    max_acceleration = df['total_acceleration'].max() if 'total_acceleration' in df.columns else 0
    
    # Calculate average gyroscope magnitude
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

def create_kpi_card(label: str, value: str, icon: str = "📊") -> str:
    """Create a KPI card HTML"""
    return f"""
    <div class="kpi-card">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.2rem; margin-right: 8px;">{icon}</span>
            <span class="kpi-label">{label}</span>
        </div>
        <div class="kpi-value">{value}</div>
    </div>
    """

def render_kpi_dashboard(kpis: Dict[str, float]):
    """Render improved KPI dashboard with responsive layout"""
    st.subheader("📊 Performance Dashboard")
    
    # Create responsive KPI grid
    kpi_data = [
        ("Distance", f"{kpis['total_distance_km']:.2f} km", "🛣️"),
        ("Max Speed", f"{kpis['max_speed_ms']:.1f} m/s", "⚡"),
        ("Avg Speed", f"{kpis['avg_speed_ms']:.1f} m/s", "🏃"),
        ("Energy", f"{kpis['total_energy_mj']:.2f} MJ", "🔋"),
        ("Avg Power", f"{kpis['avg_power_w']:.1f} W", "💡"),
        ("Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ", "♻️"),
        ("Max Accel", f"{kpis['max_acceleration']:.2f} m/s²", "📈"),
        ("Avg Gyro", f"{kpis['avg_gyro_magnitude']:.2f} °/s", "🎯")
    ]
    
    # Create columns for responsive layout
    num_cols = min(4, len(kpi_data))  # Maximum 4 columns
    cols = st.columns(num_cols)
    
    for i, (label, value, icon) in enumerate(kpi_data):
        with cols[i % num_cols]:
            st.metric(
                label=label,
                value=value,
                border=True,
                help=f"Current {label.lower()} measurement"
            )

def create_speed_chart(df: pd.DataFrame):
    """Create speed over time chart"""
    if df.empty or 'speed_ms' not in df.columns:
        return go.Figure().add_annotation(
            text="No speed data available", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.line(
        df, x='timestamp', y='speed_ms',
        title='Vehicle Speed Over Time',
        labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'}
    )
    fig.update_layout(height=400)
    return fig

def create_power_chart(df: pd.DataFrame):
    """Create power system chart"""
    if df.empty or not all(col in df.columns for col in ['voltage_v', 'current_a', 'power_w']):
        return go.Figure().add_annotation(
            text="No power data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=('Voltage & Current', 'Power')
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['voltage_v'], 
                  name='Voltage (V)', line=dict(color='blue')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['current_a'], 
                  name='Current (A)', line=dict(color='red')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_w'], 
                  name='Power (W)', line=dict(color='green')), 
        row=2, col=1
    )
    
    fig.update_layout(height=500, title_text="Electrical System Performance")
    return fig

def create_imu_chart(df: pd.DataFrame):
    """Create IMU (gyroscope and accelerometer) analysis chart"""
    if df.empty or not all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']):
        return go.Figure().add_annotation(
            text="No IMU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Gyroscope Data (deg/s)', 'Accelerometer Data (m/s²)'),
        vertical_spacing=0.1
    )
    
    # Gyroscope data
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gyro_x'], 
                  name='Gyro X', line=dict(color='red')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gyro_y'], 
                  name='Gyro Y', line=dict(color='green')), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gyro_z'], 
                  name='Gyro Z', line=dict(color='blue')), 
        row=1, col=1
    )
    
    # Accelerometer data
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['accel_x'], 
                  name='Accel X', line=dict(color='orange')), 
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['accel_y'], 
                  name='Accel Y', line=dict(color='purple')), 
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['accel_z'], 
                  name='Accel Z', line=dict(color='brown')), 
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="IMU Sensor Data")
    return fig

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
        title='Efficiency Analysis: Speed vs Power Consumption',
        labels={'speed_ms': 'Speed (m/s)', 'power_w': 'Power (W)'}
    )
    fig.update_layout(height=400)
    return fig

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
    
    # Calculate the center point for the map view
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
        title='Vehicle Track and Performance',
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
    # Remove timestamp-related columns
    exclude_cols = ['message_id', 'uptime_seconds']
    return [col for col in numeric_columns if col not in exclude_cols]

def create_dynamic_chart(df: pd.DataFrame, chart_config: Dict[str, Any]):
    """Create a dynamic chart based on user configuration"""
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
        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'bar':
            # For bar charts, we'll use recent data points
            recent_df = df.tail(20)
            fig = px.bar(recent_df, x=x_col, y=y_col, title=title)
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=y_col, title=f'Distribution of {y_col}')
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title)
        
        fig.update_layout(height=400)
        return fig
    
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

# Updated dynamic charts section with improved scroll position preservation
@st.fragment(run_every="3s")
def render_dynamic_charts_section(df: pd.DataFrame):
    """Render the dynamic charts section as a fragment with scroll preservation"""
    
    # Mark this as an auto-refresh to handle scroll differently
    st.session_state.is_auto_refresh = True
    
    # Create a container to prevent accumulation
    with st.container():
        st.subheader("📊 Dynamic Charts")
        
        # Get available columns with error handling
        try:
            available_columns = get_available_columns(df)
        except Exception as e:
            st.error(f"Error getting available columns: {e}")
            available_columns = []
        
        if not available_columns:
            st.warning("No numeric data available for creating charts.")
            return
        
        # Controls container
        with st.container():
            # Add chart button with error handling
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("➕ Add Chart", key="add_chart_btn", help="Create a new custom chart"):
                    try:
                        # Create new chart configuration
                        new_chart = {
                            'id': str(uuid.uuid4()),
                            'title': 'New Chart',
                            'chart_type': 'line',
                            'x_axis': 'timestamp' if 'timestamp' in df.columns else available_columns[0],
                            'y_axis': available_columns[0] if available_columns else None
                        }
                        st.session_state.dynamic_charts.append(new_chart)
                        st.session_state.is_auto_refresh = False  # Manual action, allow scroll jump
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding chart: {e}")
            
            with col2:
                if st.session_state.dynamic_charts:
                    st.info(f"📈 {len(st.session_state.dynamic_charts)} custom chart(s) created")
        
        # Charts container
        with st.container():
            # Display existing charts with improved error handling
            if st.session_state.dynamic_charts:
                for i, chart_config in enumerate(st.session_state.dynamic_charts):
                    try:
                        with st.container(border=True, key=f"chart_container_{chart_config['id']}"):
                            st.markdown("---")
                            
                            # Chart configuration controls
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
                                        st.session_state.is_auto_refresh = False  # Manual action, allow scroll jump
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting chart: {e}")
                            
                            # Display the chart with error handling
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
    """Main dashboard function"""
    st.markdown(
        '<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', 
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    
    # Sidebar - Connection Management
    st.sidebar.header("🔗 Connection Management")
    
    # Connection controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Connect"):
            if st.session_state.subscriber:
                st.session_state.subscriber.disconnect()
                time.sleep(2)
            
            with st.spinner("Connecting to Ably..."):
                st.session_state.subscriber = TelemetrySubscriber()
                if st.session_state.subscriber.connect():
                    st.sidebar.success("✅ Connected!")
                else:
                    st.sidebar.error("❌ Connection failed!")
            
            st.session_state.is_auto_refresh = False  # Manual action, allow scroll jump
            st.rerun()
    
    with col2:
        if st.button("🛑 Disconnect"):
            if st.session_state.subscriber:
                st.session_state.subscriber.disconnect()
                st.session_state.subscriber = None
            st.sidebar.info("🛑 Disconnected")
            st.session_state.is_auto_refresh = False  # Manual action, allow scroll jump
            st.rerun()
    
    # Connection status and stats
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        st.sidebar.markdown(
            '<div class="connection-status status-connected">✅ Connected</div>',
            unsafe_allow_html=True
        )
        stats = st.session_state.subscriber.get_stats()
    else:
        st.sidebar.markdown(
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
    
    # Display stats
    col1, col2 = st.sidebar.columns(2)
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
        st.sidebar.error(f"Last Error: {stats['last_error'][:50]}...")
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    new_auto_refresh = st.sidebar.checkbox(
        "Auto Refresh", 
        value=st.session_state.auto_refresh
    )
    
    # Track if auto-refresh setting changed
    if new_auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = new_auto_refresh
        st.session_state.is_auto_refresh = False  # Manual change, allow scroll jump
    
    if st.session_state.auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
    
    st.sidebar.info(f"📡 Channel: {CHANNEL_NAME}")
    
    # Get new messages and update data
    new_messages_count = 0
    if st.session_state.subscriber and st.session_state.subscriber.is_connected:
        new_messages = st.session_state.subscriber.get_messages()
        
        if new_messages:
            new_messages_count = len(new_messages)
            new_df = pd.DataFrame(new_messages)
            
            # Process timestamps
            if 'timestamp' in new_df.columns:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            
            # Append to existing data
            if st.session_state.telemetry_data.empty:
                st.session_state.telemetry_data = new_df
            else:
                st.session_state.telemetry_data = pd.concat([
                    st.session_state.telemetry_data, new_df
                ], ignore_index=True)
            
            # Keep only recent data
            if len(st.session_state.telemetry_data) > MAX_DATAPOINTS:
                st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_DATAPOINTS)
            
            st.session_state.last_update = datetime.now()
    
    if new_messages_count > 0:
        st.sidebar.success(f"📨 +{new_messages_count} new messages")
    
    # Main content
    df = st.session_state.telemetry_data.copy()
    
    if df.empty:
        st.warning("⏳ Waiting for telemetry data...")
        st.info("1. Make sure maindata.py is running\n2. Click 'Connect' to start receiving data")
        
        with st.expander("🔍 Debug Info"):
            st.json({
                "Connected": st.session_state.subscriber.is_connected if st.session_state.subscriber else False,
                "Messages": stats['messages_received'],
                "Errors": stats['errors'],
                "Channel": CHANNEL_NAME,
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set"
            })
    else:
        # KPIs - Using improved layout
        kpis = calculate_kpis(df)
        render_kpi_dashboard(kpis)
        
        st.info(f"📊 {len(df)} data points | Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Charts with preserved tab state
        st.subheader("📈 Real-time Analytics")
        
        # Use session state to remember active tab and prevent scroll jumping
        tab_names = ["Speed Analysis", "Power System", "IMU Sensors", "Efficiency", "GPS Track", "Dynamic Charts", "Raw Data"]
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            st.plotly_chart(create_speed_chart(df), use_container_width=True)
        
        with tabs[1]:
            st.plotly_chart(create_power_chart(df), use_container_width=True)
        
        with tabs[2]:
            st.plotly_chart(create_imu_chart(df), use_container_width=True)
        
        with tabs[3]:
            st.plotly_chart(create_efficiency_chart(df), use_container_width=True)
        
        with tabs[4]:
            st.plotly_chart(create_gps_map(df), use_container_width=True)
        
        with tabs[5]:
            # Use the improved dynamic charts section
            render_dynamic_charts_section(df)
        
        with tabs[6]:
            st.subheader("Raw Telemetry Data")
            st.dataframe(df.tail(100), use_container_width=True)
            
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
    
    # Auto-refresh with scroll position preservation
    if st.session_state.auto_refresh and st.session_state.subscriber and st.session_state.subscriber.is_connected:
        # Only sleep and rerun if this is not from a fragment auto-refresh
        if not hasattr(st.session_state, 'fragment_rerun') or not st.session_state.fragment_rerun:
            time.sleep(refresh_interval)
            st.session_state.is_auto_refresh = True  # Mark as auto-refresh to preserve scroll
            st.rerun()
    
    # Reset auto-refresh flag after each run
    st.session_state.is_auto_refresh = False
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<p>Shell Eco-marathon Telemetry Dashboard | Real-time Data Visualization with IMU Integration</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
