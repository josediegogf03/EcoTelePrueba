import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import logging
import asyncio
import threading
from queue import Queue, Empty
import os

# Ably import with fallback handling
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    st.error("Ably library not installed. Please install with: pip install ably")

# Configure page
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b35;
    }
    .connection-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .connecting {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
ABLY_API_KEY = os.environ.get('ABLY_API_KEY', ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATA_POINTS = 1000  # Maximum data points to keep in memory

class AblyTelemetrySubscriber:
    def __init__(self):
        self.realtime = None
        self.channel = None
        self.is_connected = False
        self.connection_status = "disconnected"
        self.data_queue = Queue()
        self.total_messages_received = 0
        self.last_message_time = None
        self.connection_thread = None
        self.message_thread = None
        self._stop_event = threading.Event()
        
    def _validate_api_key(self) -> bool:
        """Validate Ably API key format and availability"""
        if not ABLY_API_KEY:
            return False
        
        if len(ABLY_API_KEY) < 20:
            return False
        
        if "YOUR_ACTUAL_API_KEY_PLACEHOLDER" in ABLY_API_KEY:
            return False
        
        return True
    
    def _on_message(self, message):
        """Handle incoming messages"""
        try:
            data = message.data
            if isinstance(data, str):
                data = json.loads(data)
            
            # Add reception timestamp
            data['reception_timestamp'] = datetime.now().isoformat()
            
            # Put data in queue for main thread to process
            self.data_queue.put(data)
            self.total_messages_received += 1
            self.last_message_time = datetime.now()
            
        except Exception as e:
            st.error(f"Error processing message: {e}")
    
    def _on_connection_state_change(self, state_change=None):
        """Handle connection state changes"""
        if state_change:
            current_state = getattr(state_change, 'current', 'unknown')
            self.connection_status = current_state
            self.is_connected = current_state == 'connected'
        
    def connect(self):
        """Connect to Ably in a separate thread"""
        if not self._validate_api_key():
            st.error("Invalid Ably API key")
            return False
        
        if self.connection_thread and self.connection_thread.is_alive():
            return True
        
        self.connection_status = "connecting"
        self._stop_event.clear()
        self.connection_thread = threading.Thread(target=self._connect_worker)
        self.connection_thread.daemon = True
        self.connection_thread.start()
        
        return True
    
    def _connect_worker(self):
        """Connection worker that runs in separate thread"""
        try:
            # Create AblyRealtime instance
            self.realtime = AblyRealtime(ABLY_API_KEY)
            
            # Set up connection state listeners with fallback handling
            try:
                self.realtime.connection.on('connected', self._on_connection_state_change)
                self.realtime.connection.on('disconnected', self._on_connection_state_change)
                self.realtime.connection.on('suspended', self._on_connection_state_change)
                self.realtime.connection.on('failed', self._on_connection_state_change)
            except (AttributeError, TypeError):
                # Fallback for versions that don't support event listeners
                pass
            
            # Wait for connection with timeout and polling
            timeout = 15.0
            start_time = time.time()
            
            while time.time() - start_time < timeout and not self._stop_event.is_set():
                try:
                    # Check connection state
                    if hasattr(self.realtime.connection, 'state'):
                        current_state = self.realtime.connection.state
                        self.connection_status = current_state
                        
                        if current_state == 'connected':
                            self.is_connected = True
                            break
                        elif current_state in ['failed', 'suspended']:
                            raise Exception(f"Connection failed with state: {current_state}")
                    
                    time.sleep(0.5)  # Poll every 500ms
                    
                except Exception as e:
                    raise Exception(f"Connection check failed: {e}")
            
            if not self.is_connected:
                raise Exception("Connection timeout")
            
            # Get channel and set up message listener
            self.channel = self.realtime.channels.get(TELEMETRY_CHANNEL_NAME)
            
            # Subscribe to messages with error handling
            try:
                self.channel.subscribe('telemetry_update', self._on_message)
            except Exception as e:
                # Try alternative subscription method
                try:
                    self.channel.on('telemetry_update', self._on_message)
                except Exception as e2:
                    raise Exception(f"Failed to subscribe to messages: {e}, {e2}")
            
            # Start message processing thread
            self.message_thread = threading.Thread(target=self._message_processor)
            self.message_thread.daemon = True
            self.message_thread.start()
            
        except Exception as e:
            self.connection_status = "failed"
            self.is_connected = False
            st.error(f"Connection failed: {e}")
    
    def _message_processor(self):
        """Process messages from queue"""
        while not self._stop_event.is_set():
            try:
                # Process any pending messages
                while True:
                    try:
                        message = self.data_queue.get_nowait()
                        # Store message in session state
                        if 'telemetry_messages' not in st.session_state:
                            st.session_state.telemetry_messages = []
                        
                        st.session_state.telemetry_messages.append(message)
                        
                        # Keep only the last MAX_DATA_POINTS messages
                        if len(st.session_state.telemetry_messages) > MAX_DATA_POINTS:
                            st.session_state.telemetry_messages = st.session_state.telemetry_messages[-MAX_DATA_POINTS:]
                        
                    except Empty:
                        break
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                st.error(f"Error in message processor: {e}")
                time.sleep(1)
    
    def disconnect(self):
        """Disconnect from Ably"""
        self._stop_event.set()
        self.is_connected = False
        self.connection_status = "disconnected"
        
        if self.realtime:
            try:
                if hasattr(self.realtime, 'close'):
                    self.realtime.close()
            except Exception as e:
                pass
        
        self.realtime = None
        self.channel = None
    
    def get_connection_info(self):
        """Get current connection information"""
        return {
            'status': self.connection_status,
            'is_connected': self.is_connected,
            'total_messages': self.total_messages_received,
            'last_message': self.last_message_time,
            'queue_size': self.data_queue.qsize()
        }

def initialize_subscriber():
    """Initialize the Ably subscriber in session state"""
    if 'ably_subscriber' not in st.session_state:
        st.session_state.ably_subscriber = AblyTelemetrySubscriber()
    if 'telemetry_messages' not in st.session_state:
        st.session_state.telemetry_messages = []

def convert_messages_to_dataframe(messages):
    """Convert telemetry messages to DataFrame"""
    if not messages:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(messages)
        
        # Convert timestamp columns to datetime
        for col in ['timestamp', 'reception_timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error converting data: {e}")
        return pd.DataFrame()

def calculate_kpis(df):
    """Calculate key performance indicators"""
    if df.empty:
        return {
            'total_energy_mj': 0, 'max_speed_ms': 0, 'avg_speed_ms': 0,
            'total_distance_km': 0, 'avg_power_w': 0, 'efficiency_km_per_mj': 0
        }
    
    total_energy = df['energy_j'].sum() / 1000000 if 'energy_j' in df.columns else 0
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

def create_speed_chart(df):
    """Create speed over time chart"""
    if df.empty or 'speed_ms' not in df.columns:
        return go.Figure().add_annotation(text="No speed data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.line(df, x='timestamp', y='speed_ms', 
                  title='Vehicle Speed Over Time',
                  labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'})
    fig.update_layout(height=400)
    return fig

def create_power_chart(df):
    """Create power consumption chart"""
    if df.empty or 'voltage_v' not in df.columns:
        return go.Figure().add_annotation(text="No power data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['voltage_v'], name='Voltage (V)'),
        secondary_y=False,
    )
    
    if 'current_a' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['current_a'], name='Current (A)'),
            secondary_y=True,
        )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Voltage (V)", secondary_y=False)
    fig.update_yaxes(title_text="Current (A)", secondary_y=True)
    fig.update_layout(title_text="Electrical Parameters Over Time", height=400)
    
    return fig

def create_gps_map(df):
    """Create GPS tracking map"""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return go.Figure().add_annotation(text="No GPS data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Filter out invalid coordinates
    df_valid = df[(df['latitude'].notna()) & (df['longitude'].notna()) &
                  (df['latitude'] != 0) & (df['longitude'] != 0)]
    
    if df_valid.empty:
        return go.Figure().add_annotation(text="No valid GPS coordinates", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    color_col = 'speed_ms' if 'speed_ms' in df_valid.columns else None
    size_col = 'power_w' if 'power_w' in df_valid.columns else None
    
    fig = px.scatter_mapbox(df_valid, lat='latitude', lon='longitude',
                           color=color_col, size=size_col,
                           hover_data=['speed_ms', 'power_w', 'voltage_v'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w', 'voltage_v']) else None,
                           mapbox_style='open-street-map',
                           title='Vehicle Track and Performance',
                           height=400)
    
    # Set zoom level based on data spread
    lat_center = df_valid['latitude'].mean()
    lon_center = df_valid['longitude'].mean()
    fig.update_layout(mapbox=dict(center=dict(lat=lat_center, lon=lon_center), zoom=12))
    
    return fig

def render_connection_sidebar():
    """Render the connection management sidebar"""
    st.sidebar.header("🔌 Connection Management")
    
    subscriber = st.session_state.ably_subscriber
    conn_info = subscriber.get_connection_info()
    
    # Connection status
    status_class = "connected" if conn_info['is_connected'] else ("connecting" if conn_info['status'] == "connecting" else "disconnected")
    status_text = conn_info['status'].title()
    
    st.sidebar.markdown(f"""
    <div class="connection-status {status_class}">
        Status: {status_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Connection metrics
    st.sidebar.metric("Messages Received", conn_info['total_messages'])
    
    if conn_info['last_message']:
        time_since = datetime.now() - conn_info['last_message']
        st.sidebar.metric("Last Message", f"{time_since.total_seconds():.1f}s ago")
    else:
        st.sidebar.metric("Last Message", "Never")
    
    st.sidebar.metric("Queue Size", conn_info['queue_size'])
    
    # Connection controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Reconnect", disabled=conn_info['status'] == "connecting"):
            subscriber.disconnect()
            time.sleep(1)  # Brief delay
            subscriber.connect()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Disconnect", disabled=not conn_info['is_connected']):
            subscriber.disconnect()
            st.rerun()
    
    # Auto-connect option
    auto_connect = st.sidebar.checkbox("Auto-connect on startup", value=True)
    
    if auto_connect and not conn_info['is_connected'] and conn_info['status'] != "connecting":
        subscriber.connect()
        st.rerun()
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=True)
    
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 2)
        
        # Auto-refresh implementation
        if conn_info['is_connected']:
            time.sleep(refresh_rate)
            st.rerun()

def main():
    # Initialize subscriber
    initialize_subscriber()
    
    if not ABLY_AVAILABLE:
        st.error("Ably library not available. Please install it to use real-time features.")
        return
    
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Render connection sidebar
    render_connection_sidebar()
    
    # Main dashboard controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data display options
    max_points_display = st.sidebar.slider("Max data points to display", 50, 1000, 200)
    
    # Get telemetry data
    messages = st.session_state.telemetry_messages[-max_points_display:] if st.session_state.telemetry_messages else []
    df = convert_messages_to_dataframe(messages)
    
    # Show data status
    st.sidebar.info(f"Displaying {len(df)} data points")
    
    if df.empty:
        st.warning("No telemetry data available. Please ensure the connection is established and the data publisher is running.")
        
        # Show connection instructions
        st.info("""
        **To receive data:**
        1. Ensure the Ably API key is correctly configured
        2. Run the maindata.py script to start publishing telemetry data
        3. Check the connection status in the sidebar
        4. Enable auto-connect and auto-refresh for continuous updates
        """)
        return
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Display KPIs
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Distance", f"{kpis['total_distance_km']:.2f} km")
    with col2:
        st.metric("Max Speed", f"{kpis['max_speed_ms']:.1f} m/s")
    with col3:
        st.metric("Avg Speed", f"{kpis['avg_speed_ms']:.1f} m/s")
    with col4:
        st.metric("Total Energy", f"{kpis['total_energy_mj']:.2f} MJ")
    with col5:
        st.metric("Avg Power", f"{kpis['avg_power_w']:.1f} W")
    with col6:
        st.metric("Efficiency", f"{kpis['efficiency_km_per_mj']:.2f} km/MJ")
    
    # Charts section
    st.subheader("📈 Real-time Telemetry Analytics")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"])
    
    with tab1:
        st.plotly_chart(create_speed_chart(df), use_container_width=True)
        
        if 'speed_ms' in df.columns and not df['speed_ms'].empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Speed Statistics")
                speed_stats = df['speed_ms'].describe()
                for stat, value in speed_stats.items():
                    st.write(f"**{stat.title()}:** {value:.2f} m/s")
            
            with col2:
                fig_hist = px.histogram(df, x='speed_ms', nbins=20, 
                                      title='Speed Distribution')
                st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_power_chart(df), use_container_width=True)
        
        if 'power_w' in df.columns and not df['power_w'].empty:
            col1, col2 = st.columns(2)
            with col1:
                fig_power = px.line(df, x='timestamp', y='power_w', 
                                  title='Power Consumption Over Time')
                st.plotly_chart(fig_power, use_container_width=True)
            
            with col2:
                if 'energy_j' in df.columns:
                    df['cumulative_energy'] = df['energy_j'].cumsum() / 1000000
                    fig_energy = px.line(df, x='timestamp', y='cumulative_energy',
                                       title='Cumulative Energy Consumption (MJ)')
                    st.plotly_chart(fig_energy, use_container_width=True)
    
    with tab3:
        if 'speed_ms' in df.columns and 'power_w' in df.columns:
            # Create efficiency scatter plot
            df_clean = df.dropna(subset=['speed_ms', 'power_w'])
            if not df_clean.empty:
                fig = px.scatter(df_clean, x='speed_ms', y='power_w', 
                               color='voltage_v' if 'voltage_v' in df_clean.columns else None,
                               title='Speed vs Power Consumption',
                               labels={'speed_ms': 'Speed (m/s)', 'power_w': 'Power (W)'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency insights
        st.subheader("Efficiency Insights")
        if not df.empty and 'speed_ms' in df.columns and 'power_w' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                optimal_idx = df['speed_ms'].idxmax() if not df['speed_ms'].empty else None
                if optimal_idx is not None:
                    optimal_point = df.loc[optimal_idx]
                    st.write("**Current Optimal Performance:**")
                    st.write(f"Speed: {optimal_point['speed_ms']:.2f} m/s")
                    st.write(f"Power: {optimal_point['power_w']:.2f} W")
                    if 'voltage_v' in optimal_point:
                        st.write(f"Voltage: {optimal_point['voltage_v']:.2f} V")
            
            with col2:
                correlation = df['speed_ms'].corr(df['power_w']) if len(df) > 1 else 0
                st.write(f"**Speed-Power Correlation:** {correlation:.3f}")
                
                avg_efficiency = kpis['efficiency_km_per_mj']
                st.write(f"**Current Efficiency:** {avg_efficiency:.2f} km/MJ")
    
    with tab4:
        st.plotly_chart(create_gps_map(df), use_container_width=True)
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("Track Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Track Bounds:**")
                st.write(f"Latitude: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
                st.write(f"Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
            
            with col2:
                if len(df) > 1:
                    lat_diff = df['latitude'].max() - df['latitude'].min()
                    lon_diff = df['longitude'].max() - df['longitude'].min()
                    approx_distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111000
                    st.write(f"**Approximate Track Size:** {approx_distance:.0f} m")
    
    with tab5:
        st.subheader("Raw Telemetry Data")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"])
        with col2:
            if st.button("📥 Download CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col3:
            if st.button("🗑️ Clear Data"):
                st.session_state.telemetry_messages = []
                st.rerun()
        
        # Display data
        if show_rows == "All":
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.head(int(show_rows)), use_container_width=True)
        
        # Data summary
        if not df.empty:
            st.subheader("Data Summary")
            st.write(df.describe())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Shell Eco-marathon Telemetry Dashboard | Real-time Data via Ably WebSocket</p>
        <p>Data and Telemetry Award sponsored by Shmid Elektronik</p>
        <p>Inspiring future energy solutions through student innovation 🌱</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
