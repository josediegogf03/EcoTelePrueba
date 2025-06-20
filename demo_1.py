import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import time
from datetime import datetime, timedelta

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
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def simulate_telemetry_data(num_points=100, real_time=False):
    """
    Simulate telemetry data for Shell Eco-marathon vehicle
    """
    if real_time:
        # For real-time simulation, generate single data point
        current_time = datetime.now()
        
        # Base values with some realistic constraints
        speed = max(0, random.gauss(15, 3))  # Average speed around 15 m/s
        voltage = random.gauss(48, 2)  # Battery voltage around 48V
        current = max(0, random.gauss(8, 2))  # Current consumption
        power = voltage * current  # Power calculation
        
        # Simulate GPS coordinates (example location)
        lat_base, lon_base = 40.7128, -74.0060  # NYC as example
        latitude = lat_base + random.gauss(0, 0.001)
        longitude = lon_base + random.gauss(0, 0.001)
        
        return pd.DataFrame({
            'timestamp': [current_time],
            'speed_ms': [speed],
            'voltage_v': [voltage],
            'current_a': [current],
            'power_w': [power],
            'energy_j': [power * 1000],  # Simplified energy calculation
            'distance_m': [speed * 1000],  # Simplified distance
            'latitude': [latitude],
            'longitude': [longitude]
        })
    
    else:
        # Generate historical data
        start_time = datetime.now() - timedelta(hours=2)
        timestamps = [start_time + timedelta(seconds=i*5) for i in range(num_points)]
        
        # Simulate realistic telemetry patterns
        speeds = []
        voltages = []
        currents = []
        
        for i in range(num_points):
            # Simulate race patterns - start slow, accelerate, maintain, slow down
            if i < num_points * 0.2:  # Start phase
                speed = random.gauss(5, 1)
            elif i < num_points * 0.8:  # Main race phase
                speed = random.gauss(18, 4)
            else:  # End phase
                speed = random.gauss(8, 2)
            
            speeds.append(max(0, speed))
            
            # Voltage decreases over time (battery drain)
            voltage = 48 - (i / num_points) * 5 + random.gauss(0, 0.5)
            voltages.append(max(30, voltage))
            
            # Current varies with speed and efficiency
            current = max(0, speeds[i] * 0.5 + random.gauss(0, 1))
            currents.append(current)
        
        powers = [v * c for v, c in zip(voltages, currents)]
        energies = [p * 1000 for p in powers]  # Simplified energy
        distances = np.cumsum([s * 5 for s in speeds])  # Distance traveled
        
        # GPS simulation (moving along a track)
        lat_base, lon_base = 40.7128, -74.0060
        latitudes = [lat_base + i * 0.0001 + random.gauss(0, 0.00005) for i in range(num_points)]
        longitudes = [lon_base + i * 0.0001 + random.gauss(0, 0.00005) for i in range(num_points)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'speed_ms': speeds,
            'voltage_v': voltages,
            'current_a': currents,
            'power_w': powers,
            'energy_j': energies,
            'distance_m': distances,
            'latitude': latitudes,
            'longitude': longitudes
        })

def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_energy = df['energy_j'].sum() / 1000000  # Convert to MJ
    max_speed = df['speed_ms'].max()
    avg_speed = df['speed_ms'].mean()
    total_distance = df['distance_m'].max() / 1000  # Convert to km
    avg_power = df['power_w'].mean()
    efficiency = total_distance / total_energy if total_energy > 0 else 0  # km/MJ
    
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
    fig = px.line(df, x='timestamp', y='speed_ms', 
                  title='Vehicle Speed Over Time',
                  labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'})
    fig.update_layout(height=400)
    return fig

def create_power_chart(df):
    """Create power consumption chart"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['voltage_v'], name='Voltage (V)'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['current_a'], name='Current (A)'),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Voltage (V)", secondary_y=False)
    fig.update_yaxes(title_text="Current (A)", secondary_y=True)
    fig.update_layout(title_text="Electrical Parameters Over Time", height=400)
    
    return fig

def create_efficiency_chart(df):
    """Create efficiency analysis chart"""
    df['efficiency'] = df['distance_m'] / (df['energy_j'] / 1000000)  # m/MJ
    df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], 0)
    
    fig = px.scatter(df, x='speed_ms', y='efficiency', 
                     color='power_w', size='voltage_v',
                     title='Efficiency Analysis: Speed vs Energy Efficiency',
                     labels={'speed_ms': 'Speed (m/s)', 
                            'efficiency': 'Efficiency (m/MJ)',
                            'power_w': 'Power (W)',
                            'voltage_v': 'Voltage (V)'})
    fig.update_layout(height=400)
    return fig

def create_gps_map(df):
    """Create GPS tracking map"""
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                           color='speed_ms', size='power_w',
                           hover_data=['speed_ms', 'power_w', 'voltage_v'],
                           mapbox_style='open-street-map',
                           title='Vehicle Track and Performance',
                           height=400)
    fig.update_layout(mapbox_zoom=12)
    return fig

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source:",
        ["Historical Data", "Real-time Simulation"]
    )
    
    # Data parameters
    if data_source == "Historical Data":
        num_points = st.sidebar.slider("Number of Data Points", 50, 500, 200)
        auto_refresh = False
    else:
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
    
    # Initialize session state for real-time data
    if 'telemetry_data' not in st.session_state:
        st.session_state.telemetry_data = pd.DataFrame()
    
    # Data generation
    if data_source == "Historical Data":
        df = simulate_telemetry_data(num_points=num_points, real_time=False)
    else:
        # Real-time simulation
        if auto_refresh:
            placeholder = st.empty()
            
            # Auto-refresh loop
            for _ in range(1):  # Single iteration for demo
                new_data = simulate_telemetry_data(real_time=True)
                
                if len(st.session_state.telemetry_data) > 100:
                    # Keep only last 100 points
                    st.session_state.telemetry_data = st.session_state.telemetry_data.tail(100)
                
                st.session_state.telemetry_data = pd.concat([
                    st.session_state.telemetry_data, new_data
                ], ignore_index=True)
                
                df = st.session_state.telemetry_data.copy()
                break
        else:
            if st.sidebar.button("Generate New Data Point"):
                new_data = simulate_telemetry_data(real_time=True)
                st.session_state.telemetry_data = pd.concat([
                    st.session_state.telemetry_data, new_data
                ], ignore_index=True)
            
            df = st.session_state.telemetry_data.copy() if not st.session_state.telemetry_data.empty else simulate_telemetry_data(num_points=50)
    
    if df.empty:
        st.warning("No data available. Please generate some data first.")
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
    st.subheader("📈 Telemetry Analytics")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"])
    
    with tab1:
        st.plotly_chart(create_speed_chart(df), use_container_width=True)
        
        # Speed statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Speed Statistics")
            speed_stats = df['speed_ms'].describe()
            for stat, value in speed_stats.items():
                st.write(f"**{stat.title()}:** {value:.2f} m/s")
        
        with col2:
            # Speed distribution
            fig_hist = px.histogram(df, x='speed_ms', nbins=20, 
                                  title='Speed Distribution')
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_power_chart(df), use_container_width=True)
        
        # Power analysis
        col1, col2 = st.columns(2)
        with col1:
            # Power over time
            fig_power = px.line(df, x='timestamp', y='power_w', 
                              title='Power Consumption Over Time')
            st.plotly_chart(fig_power, use_container_width=True)
        
        with col2:
            # Energy consumption
            df['cumulative_energy'] = df['energy_j'].cumsum() / 1000000  # MJ
            fig_energy = px.line(df, x='timestamp', y='cumulative_energy',
                               title='Cumulative Energy Consumption (MJ)')
            st.plotly_chart(fig_energy, use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_efficiency_chart(df), use_container_width=True)
        
        # Efficiency insights
        st.subheader("Efficiency Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Best efficiency points
            df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
            if not df_clean.empty:
                best_efficiency = df_clean.loc[df_clean['speed_ms'].idxmax()]
                st.write("**Optimal Performance Point:**")
                st.write(f"Speed: {best_efficiency['speed_ms']:.2f} m/s")
                st.write(f"Power: {best_efficiency['power_w']:.2f} W")
                st.write(f"Voltage: {best_efficiency['voltage_v']:.2f} V")
        
        with col2:
            # Power vs Speed correlation
            if len(df) > 1:
                correlation = df['speed_ms'].corr(df['power_w'])
                st.write(f"**Speed-Power Correlation:** {correlation:.3f}")
                
                # Efficiency trend
                if 'efficiency' in df.columns:
                    avg_efficiency = df['efficiency'].replace([np.inf, -np.inf], 0).mean()
                    st.write(f"**Average Efficiency:** {avg_efficiency:.2f} m/MJ")
    
    with tab4:
        st.plotly_chart(create_gps_map(df), use_container_width=True)
        
        # GPS statistics
        st.subheader("Track Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Track Bounds:**")
            st.write(f"Latitude: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
            st.write(f"Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
        
        with col2:
            # Calculate approximate track distance
            if len(df) > 1:
                # Simplified distance calculation
                lat_diff = df['latitude'].max() - df['latitude'].min()
                lon_diff = df['longitude'].max() - df['longitude'].min()
                approx_distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111000  # rough conversion to meters
                st.write(f"**Approximate Track Length:** {approx_distance:.0f} m")
    
    with tab5:
        st.subheader("Raw Telemetry Data")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"])
        with col2:
            if st.button("Download CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Display data
        if show_rows == "All":
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.head(int(show_rows)), use_container_width=True)
        
        # Data summary
        st.subheader("Data Summary")
        st.write(df.describe())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Shell Eco-marathon Telemetry Dashboard | Data and Telemetry Award sponsored by Shmid Elektronik</p>
        <p>Inspiring future energy solutions through student innovation 🌱</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
