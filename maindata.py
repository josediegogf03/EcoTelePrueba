import asyncio
import os
import random
from datetime import datetime, timedelta
import logging

from ably import AblyRealtime
from ably.types.options import Options

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Ably Configuration ---
# Fallback to the user-provided key if environment variable is not set.
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ" # Directly embedded
ABLY_API_KEY = os.environ.get('ABLY_API_KEY', ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"

if not ABLY_API_KEY:
    logging.error("ABLY_API_KEY not found. Please set the ABLY_API_KEY environment variable or ensure fallback is present. Exiting.")
    exit()
# Check if we are using the fallback and it's not the placeholder
elif ABLY_API_KEY == ABLY_API_KEY_FALLBACK and not os.environ.get('ABLY_API_KEY'):
    if ABLY_API_KEY_FALLBACK == "YOUR_ACTUAL_API_KEY_PLACEHOLDER": # Example placeholder check
         logging.error("CRITICAL: Ably API Key is a placeholder. Please update the script or set the ABLY_API_KEY environment variable.")
         exit()
    logging.warning(f"Using fallback ABLY_API_KEY: {ABLY_API_KEY_FALLBACK[:10]}... For production, prefer setting the ABLY_API_KEY environment variable.")
else:
    logging.info(f"Using ABLY_API_KEY from environment variable: {os.environ.get('ABLY_API_KEY')[:10]}...")


# --- Data Simulation Logic ---
def simulate_telemetry_data_point():
    current_time = datetime.now()
    speed = max(0, random.gauss(15, 3))
    voltage = random.gauss(48, 2)
    current = max(0, random.gauss(8, 2))
    power = voltage * current
    lat_base, lon_base = 40.7128, -74.0060
    latitude = lat_base + random.gauss(0, 0.001)
    longitude = lon_base + random.gauss(0, 0.001)

    return {
        'timestamp': current_time.isoformat(),
        'speed_ms': speed,
        'voltage_v': voltage,
        'current_a': current,
        'power_w': power,
        'energy_j': power * 1000,
        'distance_m': speed * 1000,
        'latitude': latitude,
        'longitude': longitude
    }

# --- Main Ably Publisher Logic ---
async def main_publisher():
    logging.info(f"Initializing Ably Realtime client...")
    if not ABLY_API_KEY or ABLY_API_KEY == "YOUR_ACTUAL_API_KEY_PLACEHOLDER": # Final check before use
        logging.error("CRITICAL: Ably API Key is missing or still a placeholder value before attempting to connect. Exiting.")
        return

    try:
        options = Options(key=ABLY_API_KEY, log_level=logging.WARNING) # Quieter Ably logs
        realtime = AblyRealtime(options)
        await realtime.connection.once_async('connected')
        logging.info("Successfully connected to Ably.")
    except Exception as e:
        logging.error(f"Error connecting to Ably: {e}")
        return

    channel = realtime.channels.get(TELEMETRY_CHANNEL_NAME)
    logging.info(f"Publishing data to Ably channel: {TELEMETRY_CHANNEL_NAME}")

    try:
        while True:
            data_point = simulate_telemetry_data_point()
            try:
                await channel.publish('telemetry_update', data_point)
                logging.info(f"Published data at {data_point['timestamp']}")
            except Exception as e:
                logging.error(f"Error publishing to Ably channel: {e}")
                await asyncio.sleep(5)

            await asyncio.sleep(2)

    except KeyboardInterrupt:
        logging.info("Publisher stopping due to KeyboardInterrupt.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main_publisher: {e}")
    finally:
        if 'realtime' in locals() and realtime.connection.state in ['connected', 'connecting', 'initialized']:
            logging.info("Closing Ably connection...")
            await realtime.close()
            logging.info("Ably connection closed.")

if __name__ == "__main__":
    if not ABLY_API_KEY or ABLY_API_KEY == ABLY_API_KEY_FALLBACK and ABLY_API_KEY_FALLBACK == "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ" and not os.environ.get('ABLY_API_KEY'):
        pass # Allow fallback if env var is not set

    if not ABLY_API_KEY or (ABLY_API_KEY == "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ" and not os.environ.get('ABLY_API_KEY') and ABLY_API_KEY == "YOUR_ACTUAL_API_KEY_PLACEHOLDER"): # A bit redundant, but defensive
         logging.error("CRITICAL: Ably API Key is effectively missing or a placeholder. Please set the ABLY_API_KEY environment variable or ensure the script has a valid hardcoded key (for dev only).")
    else:
        try:
            asyncio.run(main_publisher())
        except KeyboardInterrupt:
            logging.info("maindata.py (Ably Publisher) stopped by user.")
        except Exception as e:
            logging.error(f"Unhandled exception in __main__: {e}")
