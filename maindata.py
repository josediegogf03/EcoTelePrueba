import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

# Copied from prueba2.py
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

def main_data_loop():
    print("Starting maindata.py loop...")
    output_file = "telemetry_data.csv"
    # For this version, we'll focus on real-time data generation and saving.
    # The Streamlit app (prueba3.py) will handle historical data by reading
    # a potentially larger file if maindata.py was run in a historical mode.
    # Or, maindata.py could be enhanced to take an argument for mode.
    # For now, it defaults to generating single points for real-time.

    # Accumulate data in memory before writing to file
    # This is a simple approach. For long-running processes,
    # one might append or use a database.
    all_data_df = pd.DataFrame()
    max_rows_in_memory = 150 # Keep a sliding window of data

    while True:
        try:
            new_data_point = simulate_telemetry_data(real_time=True)

            # Append new data and keep a sliding window
            all_data_df = pd.concat([all_data_df, new_data_point], ignore_index=True)
            if len(all_data_df) > max_rows_in_memory:
                all_data_df = all_data_df.tail(max_rows_in_memory)

            all_data_df.to_csv(output_file, index=False)
            print(f"{datetime.now()}: Data point generated and {output_file} updated with {len(all_data_df)} rows.")

            time.sleep(2)  # Simulate data generation interval
        except KeyboardInterrupt:
            print("maindata.py loop stopped by user.")
            break
        except Exception as e:
            print(f"Error in maindata.py loop: {e}")
            time.sleep(5) # Wait a bit before retrying on error

if __name__ == "__main__":
    main_data_loop()
