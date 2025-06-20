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
import asyncio
import tracemalloc

# Disable tracemalloc warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*tracemalloc.*")

# Enable tracemalloc to prevent the warning
tracemalloc.start()

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
        self._connection_established = threading.Event()
        self.loop = None
        self.loop_thread = None
    
    def connect(self) -> bool:
        """Connect to Ably in a separate thread with async event loop"""
        try:
            with self._lock:
                self.stats['connection_attempts'] += 1
            
            self._stop_event.clear()
            self._connection_established.clear()
            
            # Start async event loop in separate thread
            self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.loop_thread.start()
            
            # Wait for connection to establish (with timeout)
            if self._connection_established.wait(timeout=20.0):
                return self.is_connected
            else:
                with self._lock:
                    self.stats['errors'] += 1
                    self.stats['last_error'] = "Connection timeout"
                return False
            
        except Exception as e:
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
            self.is_connected = False
            return False
    
    def _run_async_loop(self):
        """Run async event loop in separate thread"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the connection coroutine
            self.loop.run_until_complete(self._async_connect())
            
        except Exception as e:
            print(f"💥 Async loop error: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = f"Async loop error: {e}"
            self.is_connected = False
            self._connection_established.set()
        finally:
            if self.loop and not self.loop.is_closed():
                self.loop.close()
    
    async def _async_connect(self):
        """Async connection worker"""
        try:
            print("🔄 Starting async connection to Ably...")
            
            # Create Ably client with proper async configuration
            client_options = {
                'key': ABLY_API_KEY,
                'auto_connect': True,
                'disconnect_on_suspend': False,
                'recover': None  # Don't recover connection
            }
            
            self.ably = AblyRealtime(client_options)
            
            # Wait for connection to be established
            connection_timeout = 15.0
            start_time = time.time()
            
            while time.time() - start_time < connection_timeout:
                try:
                    if hasattr(self.ably.connection, 'state'):
                        state = self.ably.connection.state
                        print(f"🔍 Connection state: {state}")
                        
                        if state == 'connected':
                            self.is_connected = True
                            print("✅ Connected to Ably!")
                            break
                        elif state in ['failed', 'suspended']:
                            print(f"❌ Connection failed with state: {state}")
                            with self._lock:
                                self.stats['errors'] += 1
                                self.stats['last_error'] = f"Connection state: {state}"
                            self.is_connected = False
                            self._connection_established.set()
                            return
                            
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"⚠️ Error checking connection state: {e}")
                    await asyncio.sleep(0.5)
            
            if not self.is_connected:
                print("⏰ Connection timeout")
                with self._lock:
                    self.stats['errors'] += 1
                    self.stats['last_error'] = "Connection timeout"
                self._connection_established.set()
                return
            
            # Set up channel subscription
            try:
                self.channel = self.ably.channels.get(CHANNEL_NAME)
                
                # Subscribe with proper callback
                await self.channel.subscribe('telemetry_update', self._on_message_async)
                print(f"📡 Subscribed to channel: {CHANNEL_NAME}")
                
                # Signal that connection is established
                self._connection_established.set()
                
                # Keep the connection alive and process messages
                while not self._stop_event.is_set() and self.is_connected:
                    await asyncio.sleep(1)
                    
                    # Check connection health
                    try:
                        if hasattr(self.ably.connection, 'state'):
                            current_state = self.ably.connection.state
                            if current_state not in ['connected']:
                                print(f"⚠️ Connection state changed to: {current_state}")
                                if current_state in ['failed', 'suspended', 'disconnected']:
                                    self.is_connected = False
                                    break
                    except Exception as e:
                        print(f"⚠️ Health check error: {e}")
                
            except Exception as e:
                print(f"❌ Failed to subscribe to channel: {e}")
                with self._lock:
                    self.stats['errors'] += 1
                    self.stats['last_error'] = f"Subscription error: {e}"
                self.is_connected = False
                self._connection_established.set()
                
        except Exception as e:
            print(f"💥 Async connection error: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = str(e)
            self.is_connected = False
            self._connection_established.set()
    
    async def _on_message_async(self, message):
        """Handle incoming messages asynchronously"""
        try:
            print(f"📨 Received message: {message.name}")
            
            # Extract data
            data = message.data
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    return
            
            # Validate data structure
            if not isinstance(data, dict):
                print(f"⚠️ Invalid data type: {type(data)}")
                return
            
            print(f"📊 Data keys: {list(data.keys())}")
            print(f"📊 Sample data: {dict(list(data.items())[:3])}")  # Show first 3 items
            
            # Add to queue (with size limit to prevent memory issues)
            with self._lock:
                # Clear old messages if queue is getting too full
                if self.message_queue.qsize() > 100:
                    try:
                        while self.message_queue.qsize() > 50:
                            self.message_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.message_queue.put(data)
                self.stats['messages_received'] += 1
                self.stats['last_message_time'] = datetime.now()
                
                print(f"✅ Message queued. Total received: {self.stats['messages_received']}")
            
        except Exception as e:
            print(f"❌ Message handling error: {e}")
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
            print(f"📤 Returning {len(messages)} messages to dashboard")
        
        return messages
    
    def disconnect(self):
        """Disconnect from Ably"""
        try:
            print("🛑 Disconnecting from Ably...")
            self._stop_event.set()
            self.is_connected = False
            
            # If we have an event loop, schedule the cleanup
            if self.loop and not self.loop.is_closed():
                # Use call_soon_threadsafe to schedule cleanup in the event loop
                self.loop.call_soon_threadsafe(self._schedule_cleanup)
                
                # Wait a bit for cleanup to complete
                time.sleep(2)
            
            # Close Ably connection in main thread as fallback
            if self.ably:
                try:
                    self.ably.close()
                    print("✅ Ably connection closed")
                except Exception as e:
                    print(f"⚠️ Error closing Ably: {e}")
            
            # Wait for threads to finish
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=3)
                
        except Exception as e:
            print(f"❌ Disconnect error: {e}")
            with self._lock:
                self.stats['errors'] += 1
                self.stats['last_error'] = f"Disconnect error: {e}"
        finally:
            self.ably = None
            self.channel = None
    
    def _schedule_cleanup(self):
        """Schedule async cleanup"""
        if self.loop and not self.loop.is_closed():
            asyncio.create_task(self._async_cleanup())
    
    async def _async_cleanup(self):
        """Async cleanup"""
        try:
            if self.channel:
                await self.channel.unsubscribe('telemetry_update', self._on_message_async)
                print("📡 Unsubscribed from channel")
        except Exception as e:
            print(f"⚠️ Error during async cleanup: {e}")
    
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
                time.sleep(2)  # Give more time for cleanup
            
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
                "API Key": f"{ABLY_API_KEY[:10]}..." if ABLY_API_KEY else "Not set"
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
