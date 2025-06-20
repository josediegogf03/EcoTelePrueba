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
from typing import Dict, Any, List, Optional
import threading
import queue

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
MAX_DATAPOINTS = 500

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

class TelemetrySubscriber:
    def __init__(self):
        self.ably = None
        self.channel = None
        self.is_connected = False
        self.message_queue = queue.Queue()
        self.stats = {
            'messages_received': 0,
            'last_message_time': None,
            'connection_attempts': 0,
            'errors': 0,
            'last_error': None
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connection_thread = None
    
    def connect(self) -> bool:
        """Connect to Ably using synchronous approach"""
        try:
            with self._lock:
                self.stats['connection_attempts'] += 1
            
            self._stop_event.clear()
            
            # Disconnect if already connected
            if self.ably:
                self.disconnect()
            
            # Create Ably client with synchronous approach
            self.ably = AblyRealtime(ABLY_API_KEY)
            
            # Set up connection state callbacks
            def on_connected(state_change):
                with self._lock:
                    self.is_connected = True
                print(f"✅ Ably connected: {state_change}")
            
            def on_disconnected(state_change):
                with self._lock:
                    self.is_connected = False
                print(f"❌ Ably disconnected: {state_change}")
            
            def on_failed(state_change):
                with self._lock:
                    self.is_connected = False
                    self.stats['errors'] += 1
                    self.stats['last_error'] = f"Connection failed: {state_change}"
                print(f"💥 Ably connection failed: {state_change}")
            
            # Register connection event handlers
            self.ably.connection.on('connected', on_connected)
            self.ably.connection.on('disconnected', on_disconnected)
            self.ably.connection.on('failed', on_failed)
            
            # Get channel and subscribe
            self.channel = self.ably.channels.get(CHANNEL_NAME)
            
            # Set up message handler
            def message_handler(message):
                try:
                    # Extract data
                    data = message.data
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    # Add to queue with thread safety
                    with self._lock:
                        self.message_queue.put(data)
                        self.stats['messages_received'] += 1
                        self.stats['last_message_time'] = datetime.now()
                    
                    print(f"📨 Received message #{self.stats['messages_received']}: Speed={data.get('speed_ms', 'N/A')} m/s")
                    
                except Exception as e:
                    with self._lock:
                        self.stats['errors'] += 1
                        self.stats['last_error'] = f"Message processing error: {str(e)}"
                    print(f"❌ Error processing message: {e}")
            
            # Subscribe to the channel
            self.channel.subscribe('telemetry_update', message_handler)
            
            # Wait a moment for connection to establish
            time.sleep(3)
            
            # Check if we're connected
            try:
                connection_state = getattr(self.ably.connection, 'state', 'unknown')
                print(f"🔍 Connection state: {connection_state}")
                
                if connection_state in ['connected', 'connecting']:
                    with self._lock:
                        self.is_connected = True
                    return True
                else:
                    with self._lock:
                        self.is_connected = False
                        self.stats['last_error'] = f"Connection state: {connection_state}"
                    return False
                    
            except Exception as e:
                # If we can't check state, assume connected if no errors
                print(f"⚠️ Cannot check connection state: {e}")
                with self._lock:
                    self.is_connected = True  # Optimistic assumption
                return True
            
        except Exception as e:
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
                self.is_connected = False
            print(f"💥 Connection error: {e}")
            return False
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all queued messages thread-safely"""
        messages = []
        with self._lock:
            while not self.message_queue.empty():
                try:
                    messages.append(self.message_queue.get_nowait())
                except queue.Empty:
                    break
        return messages
    
    def disconnect(self):
        """Disconnect from Ably"""
        try:
            self._stop_event.set()
            
            with self._lock:
                self.is_connected = False
            
            if self.ably:
                try:
                    # Unsubscribe from channel
                    if self.channel:
                        self.channel.unsubscribe()
                    
                    # Close connection
                    self.ably.close()
                    print("🛑 Ably connection closed")
                    
                except Exception as e:
                    print(f"⚠️ Error during disconnect: {e}")
                finally:
                    self.ably = None
                    self.channel = None
                    
        except Exception as e:
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = f"Disconnect error: {str(e)}"
            print(f"💥 Disconnect error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics thread-safely"""
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

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators"""
    if df.empty:
        return {
            'total_energy_mj': 0.0,
            'max_speed_ms': 0.0,
            'avg_speed_ms': 0.0,
            'total_distance_km': 0.0,
            'avg_power_w': 0.0,
            'efficiency_km_per_mj': 0.0
        }
    
    # Ensure numeric columns
    numeric_cols = ['energy_j', 'speed_ms', 'distance_m', 'power_w']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    total_energy = df['energy_j'].iloc[-1] / 1_000_000 if 'energy_j' in df.columns and len(df) > 0 else 0
    max_speed = df['speed_ms'].max() if 'speed_ms' in df.columns else 0
    avg_speed = df['speed_ms'].mean() if 'speed_ms' in df.columns else 0
    total_distance = df['distance_m'].iloc[-1] / 1000 if 'distance_m' in df.columns and len(df) > 0 else 0
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
    
    fig = px.scatter_mapbox(
        df_valid, lat='latitude', lon='longitude',
        color='speed_ms' if 'speed_ms' in df_valid.columns else None,
        size='power_w' if 'power_w' in df_valid.columns else None,
        hover_data=['speed_ms', 'power_w', 'voltage_v'] if all(col in df_valid.columns for col in ['speed_ms', 'power_w', 'voltage_v']) else None,
        mapbox_style='open-street-map',
        title='Vehicle Track and Performance',
        height=400,
        zoom=12
    )
    
    fig.update_layout(
        mapbox=dict(
            center=dict(
                lat=df_valid['latitude'].mean(),
                lon=df_valid['longitude'].mean()
            )
        )
    )
    return fig

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
                time.sleep(1)  # Give time for cleanup
            
            st.session_state.subscriber = TelemetrySubscriber()
            
            with st.spinner("Connecting to Ably..."):
                if st.session_state.subscriber.connect():
                    st.sidebar.success("✅ Connected!")
                else:
                    st.sidebar.error("❌ Connection failed!")
            
            st.rerun()
    
    with col2:
        if st.button("🛑 Disconnect"):
            if st.session_state.subscriber:
                st.session_state.subscriber.disconnect()
                st.session_state.subscriber = None
            st.sidebar.info("🛑 Disconnected")
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
    st.session_state.auto_refresh = st.sidebar.checkbox(
        "Auto Refresh", 
        value=st.session_state.auto_refresh
    )
    
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
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set",
                "Last Error": stats.get('last_error', 'None')
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
        
        st.info(f"📊 {len(df)} data points | Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
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
            
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Auto-refresh
    if st.session_state.auto_refresh and st.session_state.subscriber and st.session_state.subscriber.is_connected:
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
    main()
