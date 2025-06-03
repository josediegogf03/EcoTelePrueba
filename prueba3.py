import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import time # Required for time.sleep in the refresh loop

# Configure page
st.set_page_config(
    page_title="Shell Eco-marathon Telemetry Dashboard V3",
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

# calculate_kpis function remains the same as in prueba2.py
def calculate_kpis(df):
    if df.empty or not all(col in df.columns for col in ['energy_j', 'speed_ms', 'distance_m', 'power_w']):
        return {
            'total_energy_mj': 0, 'max_speed_ms': 0, 'avg_speed_ms': 0,
            'total_distance_km': 0, 'avg_power_w': 0, 'efficiency_km_per_mj': 0
        }
    total_energy = df['energy_j'].sum() / 1000000
    max_speed = df['speed_ms'].max()
    avg_speed = df['speed_ms'].mean()
    total_distance = df['distance_m'].max() / 1000
    avg_power = df['power_w'].mean()
    efficiency = total_distance / total_energy if total_energy > 0 else 0
    return {
        'total_energy_mj': total_energy, 'max_speed_ms': max_speed, 'avg_speed_ms': avg_speed,
        'total_distance_km': total_distance, 'avg_power_w': avg_power, 'efficiency_km_per_mj': efficiency
    }

# create_speed_chart function remains the same
def create_speed_chart(df):
    if df.empty or 'timestamp' not in df.columns or 'speed_ms' not in df.columns:
        return go.Figure().update_layout(title='Vehicle Speed Over Time (No data)', height=400)
    fig = px.line(df, x='timestamp', y='speed_ms',
                  title='Vehicle Speed Over Time',
                  labels={'speed_ms': 'Speed (m/s)', 'timestamp': 'Time'})
    fig.update_layout(height=400)
    return fig

# create_power_chart function remains the same
def create_power_chart(df):
    if df.empty or 'timestamp' not in df.columns or 'voltage_v' not in df.columns or 'current_a' not in df.columns:
        return go.Figure().update_layout(title='Electrical Parameters Over Time (No data)', height=400)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['voltage_v'], name='Voltage (V)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['current_a'], name='Current (A)'), secondary_y=True)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Voltage (V)", secondary_y=False)
    fig.update_yaxes(title_text="Current (A)", secondary_y=True)
    fig.update_layout(title_text="Electrical Parameters Over Time", height=400)
    return fig

# create_efficiency_chart function remains the same
def create_efficiency_chart(df):
    if df.empty or not all(col in df.columns for col in ['distance_m', 'energy_j', 'speed_ms', 'power_w', 'voltage_v']):
        return go.Figure().update_layout(title='Efficiency Analysis (No data)', height=400)
    df_copy = df.copy() # Avoid modifying original df
    df_copy['efficiency'] = (df_copy['distance_m'] / (df_copy['energy_j'] / 1000000)).replace([np.inf, -np.inf], 0)
    fig = px.scatter(df_copy, x='speed_ms', y='efficiency',
                     color='power_w', size='voltage_v',
                     title='Efficiency Analysis: Speed vs Energy Efficiency',
                     labels={'speed_ms': 'Speed (m/s)', 'efficiency': 'Efficiency (m/MJ)',
                             'power_w': 'Power (W)', 'voltage_v': 'Voltage (V)'})
    fig.update_layout(height=400)
    return fig

# create_gps_map function remains the same
def create_gps_map(df):
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude', 'speed_ms', 'power_w']):
        return go.Figure().update_layout(title='Vehicle Track and Performance (No data)', mapbox_style='open-street-map', height=400)
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                           color='speed_ms', size='power_w',
                           hover_data=['speed_ms', 'power_w', 'voltage_v'],
                           mapbox_style='open-street-map',
                           title='Vehicle Track and Performance',
                           height=400, zoom=10)
    # Set a default center if data is present but mapbox_zoom doesn't auto-center well
    if not df.empty:
         fig.update_layout(mapbox_center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()}, mapbox_zoom=12)
    else:
        fig.update_layout(mapbox_zoom=1) # Minimal zoom if no data
    return fig

DATA_FILE = "telemetry_data.csv"
PLACEHOLDER_COLS = ['timestamp', 'speed_ms', 'voltage_v', 'current_a', 'power_w',
                    'energy_j', 'distance_m', 'latitude', 'longitude']

def load_data_from_file():
    if not os.path.exists(DATA_FILE):
        #st.info(f"Data file '{DATA_FILE}' not found.")
        return pd.DataFrame(columns=PLACEHOLDER_COLS)
    try:
        df = pd.read_csv(DATA_FILE)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure all placeholder columns exist, add if missing (with NaNs)
        for col in PLACEHOLDER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except pd.errors.EmptyDataError:
        #st.info(f"Data file '{DATA_FILE}' is empty.")
        return pd.DataFrame(columns=PLACEHOLDER_COLS)
    except Exception as e:
        st.error(f"Error loading data from {DATA_FILE}: {e}")
        return pd.DataFrame(columns=PLACEHOLDER_COLS)

def main():
    st.markdown('<h1 class="main-header">🏎️ Shell Eco-marathon Telemetry Dashboard V3</h1>',
                unsafe_allow_html=True)

    st.sidebar.header("Dashboard Controls")
    st.sidebar.info(f"Monitoring `{DATA_FILE}` for data from `maindata.py`.")

    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2, disabled=not auto_refresh)

    df = load_data_from_file()

    if 'last_data_timestamp' not in st.session_state:
        st.session_state.last_data_timestamp = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = pd.DataFrame(columns=PLACEHOLDER_COLS)

    # Update session state data only if new data is different or loaded for the first time
    new_timestamp = df['timestamp'].max() if not df.empty and 'timestamp' in df.columns else None

    data_changed = False
    if new_timestamp != st.session_state.last_data_timestamp :
        st.session_state.current_data = df.copy()
        st.session_state.last_data_timestamp = new_timestamp
        data_changed = True
    elif df.empty and not st.session_state.current_data.empty: # Data became empty
        st.session_state.current_data = pd.DataFrame(columns=PLACEHOLDER_COLS)
        st.session_state.last_data_timestamp = None # Reset timestamp
        data_changed = True
    elif not df.empty and len(df) != len(st.session_state.current_data): # Fallback check if timestamp is same but length changed
        st.session_state.current_data = df.copy()
        st.session_state.last_data_timestamp = new_timestamp
        data_changed = True


    current_display_df = st.session_state.current_data

    if current_display_df.empty or current_display_df.dropna(how='all').empty:
        st.warning(f"No data available or `{DATA_FILE}` is empty/not found. Is `maindata.py` running?")
        st.info("""To see data:
1. Run `python maindata.py` in your terminal.
2. Ensure this app refreshes (auto-refresh is on by default).""")
        # Create a marker file to indicate this block was reached (this line was part of a previous test, removing for final version)
        # with open("no_data_marker.tmp", "w") as f:
        #     f.write("No data warning triggered")

    # Calculate and display KPIs
    kpis = calculate_kpis(current_display_df.dropna(subset=PLACEHOLDER_COLS[:-2])) # Exclude lat/lon from dropna for kpi
    st.subheader("📊 Key Performance Indicators")
    cols_kpi = st.columns(6)
    kpi_metrics = [
        ("Total Distance", f"{kpis.get('total_distance_km', 0):.2f} km"),
        ("Max Speed", f"{kpis.get('max_speed_ms', 0):.1f} m/s"),
        ("Avg Speed", f"{kpis.get('avg_speed_ms', 0):.1f} m/s"),
        ("Total Energy", f"{kpis.get('total_energy_mj', 0):.2f} MJ"),
        ("Avg Power", f"{kpis.get('avg_power_w', 0):.1f} W"),
        ("Efficiency", f"{kpis.get('efficiency_km_per_mj', 0):.2f} km/MJ")
    ]
    for i, (label, value) in enumerate(kpi_metrics):
        with cols_kpi[i]:
            st.metric(label, value)

    # Charts section
    st.subheader("📈 Telemetry Analytics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Speed Analysis", "Power System", "Efficiency", "GPS Track", "Raw Data"])

    display_df_for_charts = current_display_df.dropna(how='all').copy() # Use a copy that has fully empty rows removed

    with tab1:
        st.plotly_chart(create_speed_chart(display_df_for_charts), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Speed Statistics")
            if not display_df_for_charts.empty and 'speed_ms' in display_df_for_charts and not display_df_for_charts['speed_ms'].dropna().empty:
                speed_stats = display_df_for_charts['speed_ms'].describe()
                st.write(speed_stats)
            else:
                st.write("No speed data for statistics.")
        with col2:
            if not display_df_for_charts.empty and 'speed_ms' in display_df_for_charts and not display_df_for_charts['speed_ms'].dropna().empty:
                fig_hist = px.histogram(display_df_for_charts.dropna(subset=['speed_ms']), x='speed_ms', nbins=20, title='Speed Distribution')
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.write("No speed data for distribution.")
    with tab2:
        st.plotly_chart(create_power_chart(display_df_for_charts), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if not display_df_for_charts.empty and 'timestamp' in display_df_for_charts and 'power_w' in display_df_for_charts:
                fig_power = px.line(display_df_for_charts.dropna(subset=['timestamp', 'power_w']), x='timestamp', y='power_w', title='Power Consumption Over Time')
                st.plotly_chart(fig_power, use_container_width=True)
            else:
                st.write("No data for power consumption chart.")
        with col2:
            if not display_df_for_charts.empty and 'timestamp' in display_df_for_charts and 'energy_j' in display_df_for_charts:
                df_energy = display_df_for_charts.dropna(subset=['timestamp', 'energy_j']).copy()
                df_energy['cumulative_energy'] = df_energy['energy_j'].cumsum() / 1000000  # MJ
                fig_energy = px.line(df_energy, x='timestamp', y='cumulative_energy', title='Cumulative Energy Consumption (MJ)')
                st.plotly_chart(fig_energy, use_container_width=True)
            else:
                st.write("No data for cumulative energy chart.")
    with tab3:
        st.plotly_chart(create_efficiency_chart(display_df_for_charts), use_container_width=True)
        st.subheader("Efficiency Insights")
        # ... (Efficiency insights logic from previous attempt, ensuring it uses display_df_for_charts)
    with tab4:
        st.plotly_chart(create_gps_map(display_df_for_charts), use_container_width=True)
        # ... (GPS track info logic, ensuring it uses display_df_for_charts)
    with tab5:
        st.subheader("Raw Telemetry Data")
        st.dataframe(display_df_for_charts, use_container_width=True)
        if st.button("Download Displayed Data as CSV"):
            csv = display_df_for_charts.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name="displayed_telemetry.csv", mime="text/csv")
        st.subheader("Data Summary")
        if not display_df_for_charts.empty:
            st.write(display_df_for_charts.describe())
        else:
            st.write("No data to summarize.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'><p>Shell Eco-marathon Telemetry Dashboard V3 | Data source: maindata.py</p><p>Inspiring future energy solutions through student innovation 🌱</p></div>", unsafe_allow_html=True)

    if auto_refresh and data_changed:
        time.sleep(refresh_rate) # Give a small buffer for file ops if needed
        st.experimental_rerun()
    elif auto_refresh: # If no data change but auto-refresh is on, still schedule a rerun
        time.sleep(refresh_rate)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
