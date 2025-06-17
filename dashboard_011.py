import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import logging
import threading
import queue
import time
import json
import tracemalloc
import psutil
from typing import Dict, Any, Optional
import gc

# Enable memory tracking
tracemalloc.start()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dashboard.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ably import with error handling
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
except ImportError as e:
    ABLY_AVAILABLE = False
    logger.error(f"Ably library not available: {e}")

# Configuration
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
ABLY_API_KEY = os.environ.get('ABLY_API_KEY', ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATAPOINTS = 1000

# Page configuration
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
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
</style>
""", unsafe_allow_html=True)


def get_memory_info():
    """Get current memory usage and system memory limit"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current_memory_mb = memory_info.rss / 1024 / 1024
        
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / 1024 / 1024 / 1024
        
        return current_memory_mb, total_memory_gb
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return 0, 0


class SimpleAblySubscriber:
    """Simplified Ably subscriber following the documentation exactly"""
    
    def __init__(self, api_key: str, channel_name: str, data_queue: queue.Queue):
        self.api_key = api_key
        self.channel_name = channel_name
        self.data_queue = data_queue
        self.realtime = None
        self.channel = None
        self.is_connected = False
        self.is_running = False
        self.connection_thread = None
        self.stats = {
            'messages_received': 0,
            'connection_attempts': 0,
            'last_message_time': None,
            'errors': 0,
            'last_error': None
        }
        
    def _on_message(self, message):
        """Handle incoming messages - simplified version"""
        try:
            # Extract data from message
            data = None
            if hasattr(message, 'data'):
                data = message.data
            else:
                data = message
            
            # Handle different data formats
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {data}")
                    return
            elif isinstance(data, dict):
                pass  # Already a dict
            else:
                logger.warning(f"Unexpected data type: {type(data)}")
                return
            
            # Ensure timestamp
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            # Add to queue
            try:
                self.data_queue.put_nowait(data)
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = datetime.now()
                logger.info(f"✅ Message received #{self.stats['messages_received']}: {data.get('message_id', 'unknown')}")
            except queue.Full:
                # Remove oldest and add new
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(data)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
    
    def _on_connection_state_change(self, state_change):
        """Handle connection state changes"""
        try:
            current_state = getattr(state_change, 'current', 'unknown')
            previous_state = getattr(state_change, 'previous', 'unknown')
            
            logger.info(f"🔄 Connection state: {previous_state} → {current_state}")
            
            if str(current_state).lower() == 'connected':
                self.is_connected = True
                logger.info("✅ Ably connected successfully")
            else:
                self.is_connected = False
                if str(current_state).lower() in ['disconnected', 'suspended', 'failed']:
                    logger.warning(f"❌ Connection lost: {current_state}")
                
        except Exception as e:
            logger.error(f"Error in connection state handler: {e}")
    
    def _connect_to_ably(self):
        """Connect to Ably using the simplest method from documentation"""
        try:
            self.stats['connection_attempts'] += 1
            logger.info(f"🔗 Connecting to Ably (attempt {self.stats['connection_attempts']})...")
            
            # Use the simple string constructor as per documentation
            self.realtime = AblyRealtime(self.api_key)
            
            # Set up connection event handler
            try:
                if hasattr(self.realtime, 'connection') and hasattr(self.realtime.connection, 'on'):
                    self.realtime.connection.on(self._on_connection_state_change)
            except Exception as e:
                logger.warning(f"Could not set connection handler: {e}")
            
            # Wait for connection (simple polling)
            max_wait = 15  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    if hasattr(self.realtime, 'connection') and hasattr(self.realtime.connection, 'state'):
                        state = str(self.realtime.connection.state).lower()
                        logger.info(f"Connection state: {state}")
                        
                        if state == 'connected':
                            self.is_connected = True
                            break
                        elif state in ['failed', 'closed']:
                            logger.error(f"Connection failed with state: {state}")
                            return False
                except Exception as e:
                    logger.warning(f"Error checking connection state: {e}")
                
                time.sleep(1)
            
            if not self.is_connected:
                logger.error("❌ Connection timeout")
                return False
            
            # Get channel using the documented method
            logger.info(f"📡 Getting channel: {self.channel_name}")
            self.channel = self.realtime.channels.get(self.channel_name)
            
            # Subscribe to messages using the documented method
            logger.info("📨 Subscribing to telemetry_update events...")
            self.channel.subscribe('telemetry_update', self._on_message)
            
            logger.info("🎉 Successfully connected and subscribed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            return False
    
    def start(self):
        """Start the subscriber"""
        if self.is_running:
            logger.warning("Subscriber already running")
            return
        
        self.is_running = True
        self.connection_thread = threading.Thread(target=self._run, daemon=True)
        self.connection_thread.start()
        logger.info("🚀 Subscriber started")
    
    def _run(self):
        """Main subscriber loop"""
        while self.is_running:
            try:
                if not self.is_connected:
                    if self._connect_to_ably():
                        logger.info("✅ Connected and ready to receive messages")
                    else:
                        logger.error("❌ Failed to connect, retrying in 10 seconds...")
                        time.sleep(10)
                        continue
                
                # Keep alive and monitor connection
                time.sleep(5)
                
                # Check connection health
                if self.realtime and hasattr(self.realtime, 'connection'):
                    try:
                        if hasattr(self.realtime.connection, 'state'):
                            state = str(self.realtime.connection.state).lower()
                            if state not in ['connected', 'connecting']:
                                logger.warning(f"⚠️ Connection unhealthy: {state}")
                                self.is_connected = False
                    except Exception as e:
                        logger.warning(f"Error checking connection: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error in subscriber loop: {e}")
                self.is_connected = False
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
                time.sleep(5)
    
    def stop(self):
        """Stop the subscriber"""
        logger.info("🛑 Stopping subscriber...")
        self.is_running = False
        self.is_connected = False
        
        if self.realtime:
            try:
                self.realtime.close()
                logger.info("✅ Connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=5)
        
        logger.info("✅ Subscriber stopped")
    
    def reconnect(self):
        """Force reconnection"""
        logger.info("🔄 Forcing reconnection...")
        self.is_connected = False
        
        if self.realtime:
            try:
                self.realtime.close()
            except Exception as e:
                logger.error(f"Error during reconnection: {e}")
        
        self.realtime = None
        self.channel = None
    
    def get_stats(self):
        """Get subscriber statistics"""
        return {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'messages_received': self.stats['messages_received'],
            'connection_attempts': self.stats['connection_attempts'],
            'last_message_time': self.stats['last_message_time'],
            'errors': self.stats['errors'],
            'last_error': self.stats['last_error']
        }


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators"""
    if df.empty:
        return {k: 0.0 for k in ['total_energy_mj', 'max_speed_ms', 'avg_speed_ms', 
                                 'total_distance_km', 'avg_power_w', 'efficiency_km_per_mj']}
    
    # Ensure numeric columns
    numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate KPIs
    total_energy = df['energy_j'].sum() / 1_000_000 if 'energy_j' in df.columns else 0
    max_speed = df['speed_ms'].max() if 'speed_ms' in df.columns else 0
    avg_speed = df['speed_ms'].mean() if 'speed_ms' in df.columns else 0
    total_distance = df['distance_m'].max() / 1000 if 'distance_m' in df.columns else 0
    avg_power = df['power_w'].mean() if 'power_w' in df.columns else 0
    efficiency = total_distance / total_energy if total_energy > 0 else 0
    
    return {
        'total_energy_mj': total_energy,
        'max_speed_ms': max_speed,
        'avg_speed_ms': avg_speed,
        'total_distance_km': total_distance,
        'avg_power_w': avg_power,
        'efficiency_km_per_mj': efficiency
    }


def create_speed_chart(df: pd.DataFrame):
    """Create speed over time chart"""
    if df.empty or 'speed_ms' not in df.columns:
        return go.Figure().add_annotation(text="No speed data available", 
                                         xref="paper", yref="paper",
                                         x=0.5, y=0.5, showarrow=False)
    
    fig = px.line(df, x='timestamp', y='speed_ms',
                  title='Vehicle Speed Over Time',
                  labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'})
    fig.update_layout(height=400, showlegend=True)
    return fig


def create_power_chart(df: pd.DataFrame):
    """Create power system chart"""
    if df.empty or not all(col in df.columns for col in ['voltage_v', 'current_a', 'power_w']):
        return go.Figure().add_annotation(text="No power data available",
                                         xref="paper", yref="paper",
                                         x=0.5, y=0.5, showarrow=False)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Voltage & Current', 'Power'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Voltage and Current
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['voltage_v'], 
                            name='Voltage (V)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['current_a'], 
                            name='Current (A)', line=dict(color='red')), row=1, col=1)
    
    # Power
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['power_w'], 
                            name='Power (W)', line=dict(color='green')), row=2, col=1)
    
    fig.update_layout(height=500, title_text="Electrical System Performance")
    return fig


def create_efficiency_chart(df: pd.DataFrame):
    """Create efficiency analysis chart"""
    if df.empty or not all(col in df.columns for col in ['speed_ms', 'power_w', 'distance_m', 'energy_j']):
        return go.Figure().add_annotation(text="No efficiency data available",
                                         xref="paper", yref="paper",
                                         x=0.5, y=0.5, showarrow=False)
    
    # Calculate efficiency
    df_copy = df.copy()
    df_copy['efficiency'] = df_copy['distance_m'] / (df_copy['energy_j'] / 1_000_000)
    df_copy['efficiency'] = df_copy['efficiency'].replace([np.inf, -np.inf], 0)
    
    fig = px.scatter(df_copy, x='speed_ms', y='efficiency',
                     color='power_w', size='voltage_v',
                     title='Efficiency Analysis: Speed vs Energy Efficiency',
                     labels={'speed_ms': 'Speed (m/s)', 'efficiency': 'Efficiency (m/MJ)'})
    fig.update_layout(height=400)
    return fig


def create_gps_map(df: pd.DataFrame):
    """Create GPS tracking map"""
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude']):
        return go.Figure().add_annotation(text="No GPS data available",
                                         xref="paper", yref="paper",
                                         x=0.5, y=0.5, showarrow=False)
    
    # Filter valid GPS coordinates
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    if df_valid.empty:
        return go.Figure().add_annotation(text="No valid GPS coordinates",
                                         xref="paper", yref="paper",
                                         x=0.5, y=0.5, showarrow=False)
    
    fig = px.scatter_mapbox(df_valid, lat='latitude', lon='longitude',
                           color='speed_ms' if 'speed_ms' in df_valid.columns else None,
                           size='power_w' if 'power_w' in df_valid.columns else None,
                           hover_data=['speed_ms', 'power_w', 'voltage_v'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w', 'voltage_v']) else None,
                           mapbox_style='open-street-map',
                           title='Vehicle Track and Performance',
                           height=400,
                           zoom=12)
    
    # Center map on data
    fig.update_layout(
        mapbox=dict(
            center=dict(
                lat=df_valid['latitude'].mean(),
                lon=df_valid['longitude'].mean()
            )
        )
    )
    return fig


def initialize_session_state():
    """Initialize session state variables"""
    if 'data_queue' not in st.session_state:
        st.session_state.data_queue = queue.Queue(maxsize=MAX_DATAPOINTS)
    
    if 'subscriber' not in st.session_state:
        st.session_state.subscriber = None
    
    if 'telemetry_data' not in st.session_state:
        st.session_state.telemetry_data = pd.DataFrame()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()


def update_telemetry_data():
    """Update telemetry data from queue"""
    new_data = []
    
    # Get all available data from queue
    while True:
        try:
            data = st.session_state.data_queue.get_nowait()
            new_data.append(data)
        except queue.Empty:
            break
    
    if new_data:
        # Convert to DataFrame
        new_df = pd.DataFrame(new_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in new_df.columns:
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        
        # Append to existing data
        if st.session_state.telemetry_data.empty:
            st.session_state.telemetry_data = new_df
        else:
            st.session_state.telemetry_data = pd.concat([st.session_state.telemetry_data, new_df], ignore_index=True)
        
        # Keep only recent data
        if len(st.session_state.telemetry_data) > MAX_DATAPOINTS:
            st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_DATAPOINTS)
        
        st.session_state.last_update = datetime.now()
        logger.info(f"📊 Updated: {len(new_data)} new, {len(st.session_state.telemetry_data)} total")
        return len(new_data)
    
    return 0


def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Check Ably availability
    if not ABLY_AVAILABLE:
        st.error("❌ Ably library not available. Please install it using: pip install ably")
        st.stop()
    
    # Sidebar - Connection Management
    st.sidebar.header("🔗 Connection Management")
    
    # Initialize subscriber if not exists
    if st.session_state.subscriber is None:
        st.session_state.subscriber = SimpleAblySubscriber(
            api_key=ABLY_API_KEY,
            channel_name=TELEMETRY_CHANNEL_NAME,
            data_queue=st.session_state.data_queue
        )
        st.session_state.subscriber.start()
    
    # Get connection stats
    stats = st.session_state.subscriber.get_stats()
    
    # Connection status
    if stats['is_connected']:
        st.sidebar.markdown(
            '<div class="connection-status status-connected">✅ Connected</div>',
            unsafe_allow_html=True
        )
    elif stats['is_running']:
        st.sidebar.markdown(
            '<div class="connection-status status-connecting">⏳ Connecting...</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div class="connection-status status-disconnected">❌ Disconnected</div>',
            unsafe_allow_html=True
        )
    
    # Connection metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Messages", stats['messages_received'])
        st.metric("Attempts", stats['connection_attempts'])
    with col2:
        st.metric("Errors", stats['errors'])
        if stats['last_message_time']:
            time_since_last = datetime.now() - stats['last_message_time']
            st.metric("Last Msg", f"{time_since_last.total_seconds():.0f}s ago")
        else:
            st.metric("Last Msg", "Never")
    
    # Show last error if any
    if stats['last_error']:
        st.sidebar.error(f"Last Error: {stats['last_error'][:50]}...")
    
    # Connection controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Reconnect"):
            st.session_state.subscriber.reconnect()
            st.rerun()
    
    with col2:
        if st.button("🛑 Disconnect"):
            st.session_state.subscriber.stop()
            st.session_state.subscriber = None
            st.rerun()
    
    # Auto-refresh settings
    st.sidebar.subheader("⚙️ Settings")
    st.session_state.auto_refresh = st.sidebar.checkbox(
        "Auto Refresh", 
        value=st.session_state.auto_refresh,
        help="Automatically refresh dashboard with new data"
    )
    
    if st.session_state.auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)", 
            min_value=1, max_value=10, value=2
        )
    
    # Channel info
    st.sidebar.info(f"📡 Channel: {TELEMETRY_CHANNEL_NAME}")
    
    # Memory usage with limit
    current_mb, total_gb = get_memory_info()
    st.sidebar.text(f"Memory: {current_mb:.1f} MB / {total_gb:.1f} GB")
    
    # Update telemetry data
    new_messages = update_telemetry_data()
    if new_messages > 0:
        st.sidebar.success(f"📨 +{new_messages} new messages")
    
    # Main dashboard content
    df = st.session_state.telemetry_data.copy()
    
    if df.empty:
        st.warning("⏳ Waiting for telemetry data from the data provider...")
        st.info("Make sure maindata.py is running and publishing data to the Ably channel.")
        
        # Show debug info
        with st.expander("🔍 Debug Information"):
            st.json({
                "Connection Status": "Connected" if stats['is_connected'] else "Disconnected",
                "Messages Received": stats['messages_received'],
                "Connection Attempts": stats['connection_attempts'],
                "Errors": stats['errors'],
                "Last Error": stats['last_error'],
                "Channel": TELEMETRY_CHANNEL_NAME,
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set",
                "Subscriber Running": stats['is_running']
            })
    else:
        # KPIs
        kpis = calculate_kpis(df)
        
        st.subheader("📊 Key Performance Indicators")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Distance", f"{kpis['total_distance_km']:.2f} km")
        with col2:
            st.metric("Max Speed", f"{kpis['max_speed_ms']:.1f} m/s")
        with col3:
            st.metric("Avg Speed", f"{kpis['avg_speed_ms']:.1f} m/s")
        with col4:
            st.metric("Energy", f"{kpis['total_energy_mj']:.2f} MJ")
        with col5:
            st.metric("Avg Power", f"{kpis['avg_power_w']:.1f} W")
        with col6:
            st.metric("Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ")
        
        # Data info
        st.info(f"📊 Displaying {len(df)} data points | Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Charts
        st.subheader("📈 Real-time Analytics")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"
        ])
        
        with tab1:
            st.plotly_chart(create_speed_chart(df), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_power_chart(df), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_efficiency_chart(df), use_container_width=True)
        
        with tab4:
            st.plotly_chart(create_gps_map(df), use_container_width=True)
        
        with tab5:
            st.subheader("Raw Telemetry Data")
            st.dataframe(df.tail(100), use_container_width=True)
            
            # Download button
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<p>Shell Eco-marathon Telemetry Dashboard | Real-time Data Visualization</p>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        st.error(f"Fatal error: {e}")
    finally:
        # Cleanup
        if 'subscriber' in st.session_state and st.session_state.subscriber:
            st.session_state.subscriber.stop()
        
        # Force garbage collection
        gc.collect()
