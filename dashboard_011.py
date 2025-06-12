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
import tracemalloc
import threading
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import sys

# Enable memory tracing
tracemalloc.start()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard.log', mode='a')
    ]
)

# Try to import Ably with fallback handling
try:
    from ably import AblyRealtime
    ABLY_AVAILABLE = True
    logging.info("Ably library loaded successfully")
except ImportError as e:
    ABLY_AVAILABLE = False
    logging.error(f"Ably library not available: {e}")

# Configuration
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
ABLY_API_KEY = os.environ.get('ABLY_API_KEY', ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"
MAX_DATA_POINTS = 1000  # Maximum points to keep in memory

# Configure Streamlit page
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b35;
    }
</style>
""", unsafe_allow_html=True)

class AblyConnectionManager:
    """Manages Ably WebSocket connection with robust error handling"""
    
    def __init__(self):
        self.realtime = None
        self.channel = None
        self.is_connected = False
        self.connection_status = "Disconnected"
        self.data_queue = queue.Queue(maxsize=1000)
        self.message_count = 0
        self.connection_thread = None
        self.last_message_time = None
        self.auto_reconnect = True
        
    def _validate_api_key(self) -> bool:
        """Validate Ably API key"""
        if not ABLY_API_KEY or len(ABLY_API_KEY) < 20:
            logging.error("Invalid or missing Ably API key")
            return False
        return True
    
    def _on_message(self, message):
        """Handle incoming messages"""
        try:
            data = message.data
            if isinstance(data, str):
                data = json.loads(data)
            
            self.data_queue.put(data, block=False)
            self.message_count += 1
            self.last_message_time = datetime.now()
            
            logging.debug(f"Received message {self.message_count}: {data.get('message_id', 'unknown')}")
            
        except queue.Full:
            logging.warning("Data queue is full, dropping message")
        except Exception as e:
            logging.error(f"Error processing message: {e}")
    
    def _on_connection_state_change(self, state_change):
        """Handle connection state changes"""
        try:
            current_state = state_change.current if hasattr(state_change, 'current') else 'unknown'
            self.connection_status = current_state.title()
            self.is_connected = current_state == 'connected'
            
            logging.info(f"Connection state changed to: {current_state}")
            
            if current_state == 'connected':
                logging.info("Successfully connected to Ably")
            elif current_state in ['disconnected', 'suspended', 'failed']:
                logging.warning(f"Connection issue: {current_state}")
                if self.auto_reconnect and current_state == 'disconnected':
                    self._attempt_reconnect()
                    
        except Exception as e:
            logging.error(f"Error handling connection state change: {e}")
    
    def _attempt_reconnect(self):
        """Attempt to reconnect after a delay"""
        def reconnect_worker():
            time.sleep(5)  # Wait before reconnecting
            if not self.is_connected and self.auto_reconnect:
                logging.info("Attempting automatic reconnection...")
                self.connect()
        
        if not self.connection_thread or not self.connection_thread.is_alive():
            self.connection_thread = threading.Thread(target=reconnect_worker, daemon=True)
            self.connection_thread.start()
    
    def connect(self) -> bool:
        """Establish connection to Ably"""
        if not ABLY_AVAILABLE:
            self.connection_status = "Library Not Available"
            logging.error("Ably library not available")
            return False
        
        if not self._validate_api_key():
            self.connection_status = "Invalid API Key"
            return False
        
        try:
            self.connection_status = "Connecting"
            logging.info("Connecting to Ably...")
            
            # Clean up existing connection
            self.disconnect()
            
            # Create new connection with various compatibility options
            try:
                # Try with basic configuration first
                self.realtime = AblyRealtime(ABLY_API_KEY)
            except Exception as e:
                logging.error(f"Failed to create Ably instance: {e}")
                self.connection_status = "Connection Failed"
                return False
            
            # Set up connection state listeners with error handling
            try:
                if hasattr(self.realtime.connection, 'on'):
                    self.realtime.connection.on(self._on_connection_state_change)
                else:
                    logging.warning("Connection state listeners not supported")
                    
            except Exception as e:
                logging.warning(f"Could not set up connection listeners: {e}")
            
            # Get channel
            try:
                self.channel = self.realtime.channels.get(TELEMETRY_CHANNEL_NAME)
                
                # Subscribe to messages
                self.channel.subscribe('telemetry_update', self._on_message)
                
            except Exception as e:
                logging.error(f"Failed to set up channel: {e}")
                self.connection_status = "Channel Setup Failed"
                return False
            
            # Wait for connection with timeout
            max_wait = 10  # seconds
            wait_interval = 0.5
            waited = 0
            
            while waited < max_wait:
                try:
                    if hasattr(self.realtime.connection, 'state'):
                        state = self.realtime.connection.state
                        if state == 'connected':
                            self.is_connected = True
                            self.connection_status = "Connected"
                            logging.info(f"Successfully connected to channel: {TELEMETRY_CHANNEL_NAME}")
                            return True
                        elif state in ['failed', 'closed']:
                            self.connection_status = f"Connection {state.title()}"
                            logging.error(f"Connection failed with state: {state}")
                            return False
                    else:
                        # Fallback for versions without state property
                        time.sleep(2)
                        self.is_connected = True
                        self.connection_status = "Connected (Assumed)"
                        logging.info("Connected (state check not available)")
                        return True
                        
                except Exception as e:
                    logging.warning(f"Error checking connection state: {e}")
                
                time.sleep(wait_interval)
                waited += wait_interval
            
            self.connection_status = "Connection Timeout"
            logging.error("Connection timeout")
            return False
            
        except Exception as e:
            self.connection_status = f"Error: {str(e)[:50]}"
            logging.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Ably"""
        try:
            self.is_connected = False
            self.connection_status = "Disconnecting"
            
            if self.realtime:
                try:
                    if hasattr(self.realtime, 'close'):
                        if callable(self.realtime.close):
                            self.realtime.close()
                    elif hasattr(self.realtime.connection, 'close'):
                        if callable(self.realtime.connection.close):
                            self.realtime.connection.close()
                except Exception as e:
                    logging.warning(f"Error during disconnect: {e}")
                finally:
                    self.realtime = None
                    self.channel = None
            
            self.connection_status = "Disconnected"
            logging.info("Disconnected from Ably")
            
        except Exception as e:
            logging.error(f"Error during disconnect: {e}")
            self.connection_status = "Disconnect Error"
    
    def get_data(self) -> list:
        """Get all available data from queue"""
        data_points = []
        try:
            while not self.data_queue.empty():
                try:
                    data_points.append(self.data_queue.get_nowait())
                except queue.Empty:
                    break
        except Exception as e:
            logging.error(f"Error getting data from queue: {e}")
        
        return data_points
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'is_connected': self.is_connected,
            'status': self.connection_status,
            'message_count': self.message_count,
            'last_message_time': self.last_message_time,
            'queue_size': self.data_queue.qsize()
        }

def check_dependencies():
    """Check if all required libraries are installed"""
    required_libs = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'ably': 'ably'
    }
    
    missing_libs = []
    available_libs = []
    
    for lib_name, import_name in required_libs.items():
        try:
            __import__(import_name)
            available_libs.append(lib_name)
            logging.info(f"✓ {lib_name} is available")
        except ImportError:
            missing_libs.append(lib_name)
            logging.error(f"✗ {lib_name} is missing")
    
    if missing_libs:
        st.error(f"Missing required libraries: {', '.join(missing_libs)}")
        st.info("Please install missing libraries using: pip install " + ' '.join(missing_libs))
        return False
    else:
        logging.info("All required libraries are available")
        return True

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'connection_manager' not in st.session_state:
        st.session_state.connection_manager = AblyConnectionManager()
    
    if 'telemetry_data' not in st.session_state:
        st.session_state.telemetry_data = pd.DataFrame()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

def calculate_kpis(df):
    """Calculate key performance indicators"""
    if df.empty:
        return {
            'total_energy_mj': 0,
            'max_speed_ms': 0,
            'avg_speed_ms': 0,
            'total_distance_km': 0,
            'avg_power_w': 0,
            'efficiency_km_per_mj': 0
        }
    
    try:
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
    except Exception as e:
        logging.error(f"Error calculating KPIs: {e}")
        return {
            'total_energy_mj': 0,
            'max_speed_ms': 0,
            'avg_speed_ms': 0,
            'total_distance_km': 0,
            'avg_power_w': 0,
            'efficiency_km_per_mj': 0
        }

def create_charts(df):
    """Create all dashboard charts"""
    charts = {}
    
    if df.empty:
        # Return empty charts
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", 
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
        return {
            'speed': empty_fig,
            'power': empty_fig,
            'efficiency': empty_fig,
            'gps_map': empty_fig
        }
    
    try:
        # Speed chart
        if 'speed_ms' in df.columns and 'timestamp' in df.columns:
            charts['speed'] = px.line(df, x='timestamp', y='speed_ms',
                                    title='Vehicle Speed Over Time',
                                    labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'})
            charts['speed'].update_layout(height=400)
        
        # Power chart
        if all(col in df.columns for col in ['timestamp', 'voltage_v', 'current_a']):
            fig_power = make_subplots(specs=[[{"secondary_y": True}]])
            fig_power.add_trace(
                go.Scatter(x=df['timestamp'], y=df['voltage_v'], name='Voltage (V)'),
                secondary_y=False,
            )
            fig_power.add_trace(
                go.Scatter(x=df['timestamp'], y=df['current_a'], name='Current (A)'),
                secondary_y=True,
            )
            fig_power.update_xaxes(title_text="Time")
            fig_power.update_yaxes(title_text="Voltage (V)", secondary_y=False)
            fig_power.update_yaxes(title_text="Current (A)", secondary_y=True)
            fig_power.update_layout(title_text="Electrical Parameters Over Time", height=400)
            charts['power'] = fig_power
        
        # GPS map
        if all(col in df.columns for col in ['latitude', 'longitude', 'speed_ms']):
            charts['gps_map'] = px.scatter_mapbox(
                df, lat='latitude', lon='longitude', color='speed_ms',
                hover_data=['speed_ms', 'power_w'] if 'power_w' in df.columns else ['speed_ms'],
                mapbox_style='open-street-map',
                title='Vehicle Track and Performance',
                height=400
            )
            charts['gps_map'].update_layout(mapbox_zoom=12)
    
    except Exception as e:
        logging.error(f"Error creating charts: {e}")
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error creating charts: {str(e)}", 
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
        charts['error'] = empty_fig
    
    return charts

def render_connection_sidebar():
    """Render connection management sidebar"""
    st.sidebar.header("🔌 Connection Management")
    
    connection_manager = st.session_state.connection_manager
    status = connection_manager.get_status()
    
    # Connection status display
    status_class = "connected" if status['is_connected'] else "disconnected"
    if status['status'] == "Connecting":
        status_class = "connecting"
    
    st.sidebar.markdown(f"""
    <div class="connection-status {status_class}">
        Status: {status['status']}
    </div>
    """, unsafe_allow_html=True)
    
    # Connection metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Messages", status['message_count'])
    with col2:
        st.metric("Queue", status['queue_size'])
    
    # Last message time
    if status['last_message_time']:
        time_diff = datetime.now() - status['last_message_time']
        st.sidebar.write(f"Last message: {time_diff.total_seconds():.1f}s ago")
    else:
        st.sidebar.write("No messages received")
    
    # Connection controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔗 Connect", disabled=status['is_connected']):
            with st.spinner("Connecting..."):
                success = connection_manager.connect()
                if success:
                    st.success("Connected!")
                else:
                    st.error("Connection failed")
                st.rerun()
    
    with col2:
        if st.button("🔌 Disconnect", disabled=not status['is_connected']):
            connection_manager.disconnect()
            st.info("Disconnected")
            st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", 
                                     value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 10, 2)
        
        # Auto-refresh logic
        if status['is_connected']:
            time.sleep(refresh_rate)
            st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Connection settings
    st.sidebar.subheader("Connection Settings")
    
    # Auto-reconnect toggle
    connection_manager.auto_reconnect = st.sidebar.checkbox(
        "Auto Reconnect", value=connection_manager.auto_reconnect
    )
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.telemetry_data = pd.DataFrame()
        st.success("Data cleared")
        st.rerun()

def update_telemetry_data():
    """Update telemetry data from connection manager"""
    connection_manager = st.session_state.connection_manager
    new_data_points = connection_manager.get_data()
    
    if new_data_points:
        try:
            # Convert new data points to DataFrame
            new_df = pd.DataFrame(new_data_points)
            
            # Convert timestamp if it's a string
            if 'timestamp' in new_df.columns:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            
            # Append to existing data
            if not st.session_state.telemetry_data.empty:
                st.session_state.telemetry_data = pd.concat([
                    st.session_state.telemetry_data, new_df
                ], ignore_index=True)
            else:
                st.session_state.telemetry_data = new_df
            
            # Limit data points to prevent memory issues
            if len(st.session_state.telemetry_data) > MAX_DATA_POINTS:
                st.session_state.telemetry_data = st.session_state.telemetry_data.tail(MAX_DATA_POINTS)
            
            # Sort by timestamp if available
            if 'timestamp' in st.session_state.telemetry_data.columns:
                st.session_state.telemetry_data = st.session_state.telemetry_data.sort_values('timestamp')
            
            st.session_state.last_update = datetime.now()
            
        except Exception as e:
            logging.error(f"Error updating telemetry data: {e}")
            st.error(f"Error processing new data: {e}")

def main():
    """Main dashboard function"""
    # Check dependencies
    if not check_dependencies():
        st.stop()
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Real-time Telemetry Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Render sidebar
    render_connection_sidebar()
    
    # Update data from connection
    update_telemetry_data()
    
    # Get current data
    df = st.session_state.telemetry_data.copy()
    
    # Display connection info in main area
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        status = st.session_state.connection_manager.get_status()
        if status['is_connected']:
            st.success(f"🟢 Connected - {status['message_count']} messages received")
        else:
            st.error(f"🔴 Disconnected - Status: {status['status']}")
    
    with col2:
        st.info(f"📊 Data Points: {len(df)}")
    
    with col3:
        if st.session_state.last_update:
            time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
            st.info(f"🕒 Updated: {time_since_update:.1f}s ago")
    
    # Show data or placeholder
    if df.empty:
        st.warning("📡 No data received yet. Make sure the data publisher (maindata.py) is running and connected.")
        
        # Show connection help
        with st.expander("🔧 Connection Help"):
            st.markdown("""
            **To receive data:**
            1. Make sure the `maindata.py` script is running
            2. Verify both scripts use the same Ably API key
            3. Check that the channel name matches: `telemetry-dashboard-channel`
            4. Click the 'Connect' button in the sidebar
            
            **Troubleshooting:**
            - Check the dashboard logs for connection errors
            - Verify your internet connection
            - Ensure the Ably API key is valid
            """)
        
        st.stop()
    
    # Calculate and display KPIs
    kpis = calculate_kpis(df)
    
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
    
    # Create and display charts
    st.subheader("📈 Real-time Telemetry Analytics")
    
    charts = create_charts(df)
    
    # Chart tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"])
    
    with tab1:
        if 'speed' in charts:
            st.plotly_chart(charts['speed'], use_container_width=True)
        
        # Speed statistics
        if 'speed_ms' in df.columns:
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
        if 'power' in charts:
            st.plotly_chart(charts['power'], use_container_width=True)
        
        # Power analysis
        if 'power_w' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_power = px.line(df, x='timestamp', y='power_w', 
                                  title='Power Consumption Over Time')
                st.plotly_chart(fig_power, use_container_width=True)
            
            with col2:
                if 'energy_j' in df.columns:
                    df_copy = df.copy()
                    df_copy['cumulative_energy'] = df_copy['energy_j'].cumsum() / 1000000
                    fig_energy = px.line(df_copy, x='timestamp', y='cumulative_energy',
                                       title='Cumulative Energy Consumption (MJ)')
                    st.plotly_chart(fig_energy, use_container_width=True)
    
    with tab3:
        # Efficiency analysis
        if all(col in df.columns for col in ['speed_ms', 'power_w']):
            fig_efficiency = px.scatter(df, x='speed_ms', y='power_w',
                                      title='Speed vs Power Consumption',
                                      labels={'speed_ms': 'Speed (m/s)', 'power_w': 'Power (W)'})
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with tab4:
        if 'gps_map' in charts:
            st.plotly_chart(charts['gps_map'], use_container_width=True)
        
        # GPS statistics
        if all(col in df.columns for col in ['latitude', 'longitude']):
            st.subheader("Track Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Track Bounds:**")
                st.write(f"Latitude: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
                st.write(f"Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
            
            with col2:
                lat_range = df['latitude'].max() - df['latitude'].min()
                lon_range = df['longitude'].max() - df['longitude'].min()
                approx_distance = np.sqrt(lat_range**2 + lon_range**2) * 111000
                st.write(f"**Approximate Track Size:** {approx_distance:.0f} m")
    
    with tab5:
        st.subheader("Raw Telemetry Data")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"])
        with col2:
            if st.button("📥 Export CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col3:
            st.write(f"**Total Records:** {len(df)}")
        
        # Display data
        if show_rows == "All":
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.tail(int(show_rows)), use_container_width=True)
        
        # Data summary
        if st.checkbox("Show Data Summary"):
            st.subheader("Data Summary")
            st.write(df.describe())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Shell Eco-marathon Real-time Telemetry Dashboard | Powered by Ably WebSocket</p>
        <p>🌱 Inspiring future energy solutions through student innovation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
