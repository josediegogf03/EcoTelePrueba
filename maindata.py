import asyncio
import json
import logging
import math
import random
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import queue
import struct

try:
    from ably import AblyRealtime
except ImportError:
    print("Error: Ably library not installed. Run: pip install ably")
    sys.exit(1)

# Configuration for ESP32 MQTT source (from Transmiter.cpp)
ESP32_ABLY_API_KEY = "ja_fwQ.K6CTEw:F-aWFMdJXPCv9MvxhYztCGna3XdRJZVgA0qm9pMfDOQ"
ESP32_CHANNEL_NAME = "EcoTele"

# Configuration for Dashboard output (existing configuration)
DASHBOARD_ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
DASHBOARD_CHANNEL_NAME = "telemetry-dashboard-channel"

# Mock data configuration
MOCK_DATA_INTERVAL = 2.0  # seconds

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class TelemetryBridge:
    """
    Bridge class that either:
    1. Subscribes to ESP32 MQTT telemetry data via Ably and republishes it to the dashboard channel
    2. Generates mock telemetry data and publishes it to the dashboard channel (when mock_mode is enabled)
    """

    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        self.esp32_client = None
        self.dashboard_client = None
        self.esp32_channel = None
        self.dashboard_channel = None
        self.running = False
        self.message_queue = queue.Queue()
        self.stats = {
            "messages_received": 0,
            "messages_republished": 0,
            "last_message_time": None,
            "errors": 0,
            "last_error": None,
        }
        self.BINARY_FORMAT = "<fffffI"
        self.BINARY_FIELD_NAMES = [
            'speed_ms',
            'voltage_v',
            'current_a',
            'latitude',
            'longitude',
            'message_id',
        ]
        self.BINARY_MESSAGE_SIZE = struct.calcsize(self.BINARY_FORMAT)
        self.cumulative_distance = 0.0
        self.cumulative_energy = 0.0
        self.simulation_time = 0
        self.vehicle_heading = 0.0
        self.prev_speed = 0.0
        self.message_count = 0
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if self.mock_mode:
            logger.info(
                "🎭 MOCK MODE ENABLED - Will generate simulated telemetry data"
            )
        else:
            logger.info(
                "🔗 REAL MODE ENABLED - Will connect to ESP32 for real telemetry data"
            )

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def generate_mock_telemetry_data(self) -> Dict[str, Any]:
        current_time = datetime.now()
        base_speed = 15.0 + 5.0 * math.sin(self.simulation_time * 0.1)
        speed_variation = random.gauss(0, 1.5)
        speed = max(0, min(25, base_speed + speed_variation))
        voltage = max(40, min(55, 48.0 + random.gauss(0, 1.5)))
        current = max(0, min(15, 8.0 + speed * 0.2 + random.gauss(0, 1.0)))
        power = voltage * current
        energy_delta = power * MOCK_DATA_INTERVAL
        distance_delta = speed * MOCK_DATA_INTERVAL
        self.cumulative_energy += energy_delta
        self.cumulative_distance += distance_delta
        base_lat, base_lon = 40.7128, -74.0060
        lat_offset = 0.001 * math.sin(self.simulation_time * 0.05)
        lon_offset = 0.001 * math.cos(self.simulation_time * 0.05)
        latitude = base_lat + lat_offset + random.gauss(0, 0.0001)
        longitude = base_lon + lon_offset + random.gauss(0, 0.0001)
        turning_rate = 2.0 * math.sin(self.simulation_time * 0.08)
        gyro_x = random.gauss(0, 0.5)
        gyro_y = random.gauss(0, 0.3)
        gyro_z = turning_rate + random.gauss(0, 0.8)
        self.vehicle_heading += gyro_z * MOCK_DATA_INTERVAL
        speed_acceleration = (speed - self.prev_speed) / MOCK_DATA_INTERVAL
        self.prev_speed = speed
        accel_x = speed_acceleration + random.gauss(0, 0.2)
        accel_y = turning_rate * speed * 0.1 + random.gauss(0, 0.1)
        accel_z = 9.81 + random.gauss(0, 0.05)
        vibration_factor = speed * 0.02
        accel_x += random.gauss(0, vibration_factor)
        accel_y += random.gauss(0, vibration_factor)
        accel_z += random.gauss(0, vibration_factor)
        self.simulation_time += 1
        self.message_count += 1
        return {
            'timestamp': current_time.isoformat(),
            'speed_ms': round(speed, 2),
            'voltage_v': round(voltage, 2),
            'current_a': round(current, 2),
            'power_w': round(power, 2),
            'energy_j': round(self.cumulative_energy, 2),
            'distance_m': round(self.cumulative_distance, 2),
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'gyro_x': round(gyro_x, 3),
            'gyro_y': round(gyro_y, 3),
            'gyro_z': round(gyro_z, 3),
            'accel_x': round(accel_x, 3),
            'accel_y': round(accel_y, 3),
            'accel_z': round(accel_z, 3),
            'vehicle_heading': round(self.vehicle_heading % 360, 2),
            'total_acceleration': round(
                math.sqrt(accel_x**2 + accel_y**2 + accel_z**2), 3
            ),
            'message_id': self.message_count,
            'uptime_seconds': self.simulation_time * MOCK_DATA_INTERVAL,
            'data_source': 'MOCK_GENERATOR',
        }

    async def connect_esp32_subscriber(self) -> bool:
        if self.mock_mode:
            return True
        try:
            self.esp32_client = AblyRealtime(ESP32_ABLY_API_KEY)
            await self._wait_for_connection(self.esp32_client, "ESP32")
            self.esp32_channel = self.esp32_client.channels.get(
                ESP32_CHANNEL_NAME
            )
            await self.esp32_channel.subscribe(self._on_esp32_message_received)
            logger.info(f"✅ Connected to ESP32 channel: {ESP32_CHANNEL_NAME}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to ESP32 source: {e}")
            return False

    async def connect_dashboard_publisher(self) -> bool:
        try:
            self.dashboard_client = AblyRealtime(DASHBOARD_ABLY_API_KEY)
            await self._wait_for_connection(self.dashboard_client, "Dashboard")
            self.dashboard_channel = self.dashboard_client.channels.get(
                DASHBOARD_CHANNEL_NAME
            )
            logger.info(
                f"✅ Connected to Dashboard channel: {DASHBOARD_CHANNEL_NAME}"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Dashboard output: {e}")
            return False

    async def _wait_for_connection(self, client, name: str, timeout=10):
        logger.info(f"Waiting for {name} connection...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if client.connection.state == 'connected':
                logger.info(f"✅ {name} connection established")
                return
            await asyncio.sleep(0.1)
        logger.info(f"⏰ {name} connection timeout, assuming connected")

    def _parse_json_message(self, data_bytes: bytes) -> Optional[Dict]:
        """Parse JSON message from bytes"""
        try:
            # Handle both bytes and bytearray
            if isinstance(data_bytes, (bytes, bytearray)):
                data_str = data_bytes.decode('utf-8')
            else:
                data_str = str(data_bytes)
            
            parsed_data = json.loads(data_str)
            logger.info(f"✅ Successfully parsed JSON message with {len(parsed_data)} fields")
            return parsed_data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None

    def _parse_binary_message(self, data_bytes: bytes) -> Optional[Dict]:
        """Parse binary message from bytes"""
        if len(data_bytes) != self.BINARY_MESSAGE_SIZE:
            logger.debug(f"Binary message size mismatch. Expected {self.BINARY_MESSAGE_SIZE}, got {len(data_bytes)}")
            return None
        
        try:
            unpacked_values = struct.unpack(self.BINARY_FORMAT, data_bytes)
            data_dict = dict(zip(self.BINARY_FIELD_NAMES, unpacked_values))
            data_dict['power_w'] = data_dict['voltage_v'] * data_dict['current_a']
            logger.info(f"✅ Successfully parsed binary message")
            return data_dict
        except struct.error as e:
            logger.debug(f"Binary parsing failed: {e}")
            return None

    def _normalize_telemetry_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and enhance telemetry data for consistent dashboard format
        """
        normalized = data.copy()
        
        # Ensure we have a current timestamp if the ESP32 timestamp is invalid
        if 'timestamp' not in normalized or normalized['timestamp'].startswith('1970-01-01'):
            normalized['timestamp'] = datetime.now().isoformat()
        
        # Ensure all required fields are present with reasonable defaults
        required_fields = {
            'speed_ms': 0.0,
            'voltage_v': 0.0,
            'current_a': 0.0,
            'power_w': 0.0,
            'energy_j': 0.0,
            'distance_m': 0.0,
            'latitude': 0.0,
            'longitude': 0.0,
            'gyro_x': 0.0,
            'gyro_y': 0.0,
            'gyro_z': 0.0,
            'accel_x': 0.0,
            'accel_y': 0.0,
            'accel_z': 0.0,
            'vehicle_heading': 0.0,
            'total_acceleration': 0.0,
            'message_id': 0,
            'uptime_seconds': 0.0
        }
        
        for field, default_value in required_fields.items():
            if field not in normalized:
                normalized[field] = default_value
        
        # Calculate power if not present
        if 'power_w' not in normalized or normalized['power_w'] == 0:
            normalized['power_w'] = normalized['voltage_v'] * normalized['current_a']
        
        # Calculate total acceleration if not present
        if 'total_acceleration' not in normalized or normalized['total_acceleration'] == 0:
            normalized['total_acceleration'] = math.sqrt(
                normalized['accel_x']**2 + 
                normalized['accel_y']**2 + 
                normalized['accel_z']**2
            )
        
        # Add data source tag
        normalized['data_source'] = 'ESP32_REAL'
        
        return normalized

    def _on_esp32_message_received(self, message):
        """
        Handle incoming messages from ESP32.
        Now with improved parsing logic and data normalization.
        """
        try:
            logger.info(f"📨 Received message from ESP32 - Type: {type(message.data)}, Size: {len(message.data) if hasattr(message.data, '__len__') else 'N/A'}")
            
            data = None
            encoding = getattr(message, 'encoding', '') or ''
            
            # Try to parse the message data
            if isinstance(message.data, (bytes, bytearray)):
                # For byte data, try JSON first (most common case based on your logs)
                data = self._parse_json_message(message.data)
                
                if data is None:
                    # If JSON fails, try binary parsing
                    logger.debug("JSON parsing failed, attempting binary parsing...")
                    data = self._parse_binary_message(message.data)
                    
            elif isinstance(message.data, str):
                # Handle string data
                try:
                    data = json.loads(message.data)
                    logger.info("✅ Successfully parsed string JSON message")
                except json.JSONDecodeError:
                    logger.error("❌ Failed to parse string as JSON")
                    
            elif isinstance(message.data, dict):
                # Data is already a dictionary
                data = message.data
                logger.info("✅ Message data already in dictionary format")
                
            else:
                logger.warning(f"⚠️ Unhandled message data type: {type(message.data)}")
                return

            # Check if parsing was successful
            if data is None:
                logger.error("❌ Failed to parse message in any known format")
                logger.debug(f"Raw data preview: {str(message.data)[:200]}...")
                self.stats["errors"] += 1
                self.stats["last_error"] = "Failed to parse message"
                return

            # Normalize the data for consistent dashboard format
            normalized_data = self._normalize_telemetry_data(data)
            
            # Log successful parsing
            logger.info(
                f"📊 ESP32 Data Processed - Speed: {normalized_data.get('speed_ms', 'N/A'):.2f} m/s, "
                f"Power: {normalized_data.get('power_w', 'N/A'):.2f} W, "
                f"Voltage: {normalized_data.get('voltage_v', 'N/A'):.2f} V, "
                f"Current: {normalized_data.get('current_a', 'N/A'):.2f} A, "
                f"Msg ID: {normalized_data.get('message_id', 'N/A')}"
            )
            
            # Add to message queue for republishing
            self.message_queue.put(normalized_data)
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = datetime.now()

        except Exception as e:
            logger.error(f"❌ Error handling ESP32 message: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)

    async def generate_mock_data_loop(self):
        if not self.mock_mode:
            return
        while self.running:
            mock_data = self.generate_mock_telemetry_data()
            self.message_queue.put(mock_data)
            await asyncio.sleep(MOCK_DATA_INTERVAL)

    async def republish_messages(self):
        while self.running:
            if not self.message_queue.empty():
                messages_to_publish = []
                while (
                    not self.message_queue.empty()
                    and len(messages_to_publish) < 10
                ):
                    messages_to_publish.append(
                        self.message_queue.get_nowait()
                    )

                for message_data in messages_to_publish:
                    try:
                        await self.dashboard_channel.publish(
                            'telemetry_update', message_data
                        )
                        self.stats["messages_republished"] += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to republish message: {e}")
                        self.stats["errors"] += 1
                        self.stats["last_error"] = str(e)
                
                source_info = "MOCK" if self.mock_mode else "ESP32"
                logger.info(
                    f"📡 Republished {len(messages_to_publish)} {source_info} messages to dashboard"
                )
            await asyncio.sleep(0.1)

    async def print_stats(self):
        while self.running:
            await asyncio.sleep(30)
            mode_info = "MOCK DATA" if self.mock_mode else "ESP32 DATA"
            logger.info(
                f"📊 STATS ({mode_info}) - Received: {self.stats['messages_received']}, "
                f"Republished: {self.stats['messages_republished']}, "
                f"Errors: {self.stats['errors']}, "
                f"Queue: {self.message_queue.qsize()}"
            )
            if self.stats['last_error']:
                logger.info(f"🔍 Last Error: {self.stats['last_error']}")

    async def run(self):
        if not await self.connect_esp32_subscriber() and not self.mock_mode:
            return
        if not await self.connect_dashboard_publisher():
            return
        self.running = True
        tasks = [self.republish_messages(), self.print_stats()]
        if self.mock_mode:
            tasks.append(self.generate_mock_data_loop())
        await asyncio.gather(*tasks)

    async def cleanup(self):
        logger.info("🧹 Cleaning up...")
        if self.esp32_client:
            await self.esp32_client.close()
        if self.dashboard_client:
            await self.dashboard_client.close()


def get_user_preferences():
    """Get user preferences through terminal input"""
    print("\n" + "=" * 60)
    print("🚀 TELEMETRY BRIDGE - ESP32 to Dashboard")
    print("=" * 60)
    print()
    print("Choose data source mode:")
    print("1. 🔗 REAL DATA - Connect to ESP32 hardware")
    print("2. 🎭 MOCK DATA - Generate simulated telemetry data")
    print()

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            print("\n✅ Selected: REAL DATA mode")
            return False, 2.0
        elif choice == '2':
            print("\n✅ Selected: MOCK DATA mode")
            return True, 2.0
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


async def main():
    mock_mode, mock_interval = get_user_preferences()
    global MOCK_DATA_INTERVAL
    MOCK_DATA_INTERVAL = mock_interval
    print("\n" + "-" * 60)
    print("🔧 STARTING TELEMETRY BRIDGE...")
    print("-" * 60)
    bridge = TelemetryBridge(mock_mode=mock_mode)
    try:
        await bridge.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bridge stopped by user")
    finally:
        await bridge.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bridge interrupted")
