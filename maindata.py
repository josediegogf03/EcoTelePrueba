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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemetryBridge:
    """
    Bridge class that either:
    1. Subscribes to ESP32 MQTT telemetry data via Ably and republishes it to the dashboard channel
    2. Generates mock telemetry data and publishes it to the dashboard channel (when mock_mode is enabled)
    """
    
    def __init__(self, mock_mode=False):
        # Mode configuration
        self.mock_mode = mock_mode
        
        # Ably clients
        self.esp32_client = None
        self.dashboard_client = None
        
        # Channels
        self.esp32_channel = None
        self.dashboard_channel = None
        
        # State management
        self.running = False
        self.message_queue = queue.Queue()
        self.stats = {
            "messages_received": 0,
            "messages_republished": 0,
            "last_message_time": None,
            "errors": 0,
            "last_error": None,
        }
        
        # Mock data simulation state
        self.cumulative_distance = 0.0
        self.cumulative_energy = 0.0
        self.simulation_time = 0
        self.vehicle_heading = 0.0
        self.prev_speed = 0.0
        self.message_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Log mode
        if self.mock_mode:
            logger.info("🎭 MOCK MODE ENABLED - Will generate simulated telemetry data")
        else:
            logger.info("🔗 REAL MODE ENABLED - Will connect to ESP32 for real telemetry data")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def generate_mock_telemetry_data(self) -> Dict[str, Any]:
        """Generate realistic mock telemetry data including IMU data"""
        current_time = datetime.now()
        
        # Generate realistic speed (0-25 m/s with variations)
        base_speed = 15.0 + 5.0 * math.sin(self.simulation_time * 0.1)
        speed_variation = random.gauss(0, 1.5)
        speed = max(0, min(25, base_speed + speed_variation))
        
        # Generate electrical system data
        voltage = max(40, min(55, 48.0 + random.gauss(0, 1.5)))
        current = max(0, min(15, 8.0 + speed * 0.2 + random.gauss(0, 1.0)))
        power = voltage * current
        
        # Accumulate energy and distance
        energy_delta = power * MOCK_DATA_INTERVAL
        distance_delta = speed * MOCK_DATA_INTERVAL
        
        self.cumulative_energy += energy_delta
        self.cumulative_distance += distance_delta
        
        # Generate GPS coordinates (simulated route)
        base_lat, base_lon = 40.7128, -74.0060
        lat_offset = 0.001 * math.sin(self.simulation_time * 0.05)
        lon_offset = 0.001 * math.cos(self.simulation_time * 0.05)
        
        latitude = base_lat + lat_offset + random.gauss(0, 0.0001)
        longitude = base_lon + lon_offset + random.gauss(0, 0.0001)
        
        # Generate Gyroscope data (angular velocity in deg/s)
        turning_rate = 2.0 * math.sin(self.simulation_time * 0.08)
        gyro_x = random.gauss(0, 0.5)
        gyro_y = random.gauss(0, 0.3)
        gyro_z = turning_rate + random.gauss(0, 0.8)
        
        # Update vehicle heading
        self.vehicle_heading += gyro_z * MOCK_DATA_INTERVAL
        
        # Generate Accelerometer data (m/s²)
        speed_acceleration = (speed - self.prev_speed) / MOCK_DATA_INTERVAL
        self.prev_speed = speed
        
        accel_x = speed_acceleration + random.gauss(0, 0.2)
        accel_y = turning_rate * speed * 0.1 + random.gauss(0, 0.1)
        accel_z = 9.81 + random.gauss(0, 0.05)
        
        # Add vibration based on speed
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
            'total_acceleration': round(math.sqrt(accel_x**2 + accel_y**2 + accel_z**2), 3),
            'message_id': self.message_count,
            'uptime_seconds': self.simulation_time * MOCK_DATA_INTERVAL,
            'data_source': 'MOCK_GENERATOR'  # Identifier for mock data
        }
    
    async def connect_esp32_subscriber(self) -> bool:
        """Connect to ESP32 Ably channel to receive MQTT data"""
        if self.mock_mode:
            logger.info("🎭 Mock mode enabled - Skipping ESP32 connection")
            return True
            
        try:
            logger.info(f"Connecting to ESP32 data source...")
            logger.info(f"ESP32 API Key: {ESP32_ABLY_API_KEY[:20]}...")
            logger.info(f"ESP32 Channel: {ESP32_CHANNEL_NAME}")
            
            # Create Ably Realtime client for ESP32 data
            self.esp32_client = AblyRealtime(ESP32_ABLY_API_KEY)
            
            # Wait for connection
            await self._wait_for_connection(self.esp32_client, "ESP32")
            
            # Get channel
            self.esp32_channel = self.esp32_client.channels.get(ESP32_CHANNEL_NAME)
            
            # Subscribe to messages from ESP32
            await self.esp32_channel.subscribe(self._on_esp32_message_received)
            
            logger.info(f"✅ Connected to ESP32 channel: {ESP32_CHANNEL_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to ESP32 source: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return False
    
    async def connect_dashboard_publisher(self) -> bool:
        """Connect to Dashboard Ably channel to publish data"""
        try:
            logger.info(f"Connecting to Dashboard output...")
            logger.info(f"Dashboard API Key: {DASHBOARD_ABLY_API_KEY[:20]}...")
            logger.info(f"Dashboard Channel: {DASHBOARD_CHANNEL_NAME}")
            
            # Create Ably Realtime client for dashboard output
            self.dashboard_client = AblyRealtime(DASHBOARD_ABLY_API_KEY)
            
            # Wait for connection
            await self._wait_for_connection(self.dashboard_client, "Dashboard")
            
            # Get channel
            self.dashboard_channel = self.dashboard_client.channels.get(DASHBOARD_CHANNEL_NAME)
            
            logger.info(f"✅ Connected to Dashboard channel: {DASHBOARD_CHANNEL_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Dashboard output: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return False
    
    async def _wait_for_connection(self, client, name: str, timeout=10):
        """Wait for Ably connection to be established"""
        logger.info(f"Waiting for {name} connection...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                connection_state = client.connection.state
                if connection_state == 'connected':
                    logger.info(f"✅ {name} connection established")
                    return
                elif connection_state in ['failed', 'suspended']:
                    logger.error(f"❌ {name} connection failed: {connection_state}")
                    raise Exception(f"Connection failed with state: {connection_state}")
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"⚠️ Error checking {name} connection state: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"⏰ {name} connection timeout, assuming connected")
    
    def _on_esp32_message_received(self, message):
        """Handle incoming messages from ESP32 via Ably"""
        try:
            logger.debug(f"📨 Received message from ESP32: {message.name}")
            
            # Extract message data
            data = message.data
            
            # Parse JSON if it's a string (from MQTT it comes as binary/string)
            if isinstance(data, (str, bytes)):
                try:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                    self.stats["errors"] += 1
                    self.stats["last_error"] = f"JSON decode error: {e}"
                    return
            
            # Validate data
            if not isinstance(data, dict):
                logger.warning(f"⚠️ Invalid data type: {type(data)}")
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Invalid data type: {type(data)}"
                return
            
            # Add timestamp if not present or update it to current time
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            # Mark as real data
            data['data_source'] = 'ESP32_REAL'
            
            logger.info(f"📊 ESP32 Data received - Speed: {data.get('speed_ms', 'N/A')} m/s, "
                       f"Power: {data.get('power_w', 'N/A')} W, "
                       f"Msg ID: {data.get('message_id', 'N/A')}")
            
            # Add to queue for republishing
            self.message_queue.put(data)
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error handling ESP32 message: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = f"Message handling error: {e}"
    
    async def generate_mock_data_loop(self):
        """Generate mock telemetry data periodically"""
        if not self.mock_mode:
            return
            
        logger.info("🎭 Starting mock data generation loop")
        
        while self.running:
            try:
                # Generate mock data
                mock_data = self.generate_mock_telemetry_data()
                
                # Add to queue for republishing
                self.message_queue.put(mock_data)
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = datetime.now()
                
                # Log mock data info
                if self.message_count % 10 == 0:
                    logger.info(f"🎭 Generated MOCK message #{self.message_count} - "
                               f"Speed: {mock_data['speed_ms']} m/s, "
                               f"Power: {mock_data['power_w']:.1f} W, "
                               f"Accel: {mock_data['total_acceleration']:.2f} m/s²")
                
                # Wait for next generation cycle
                await asyncio.sleep(MOCK_DATA_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Error in mock data generation: {e}")
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Mock data generation error: {e}"
                await asyncio.sleep(1)
    
    async def republish_messages(self):
        """Process queued messages and republish them to dashboard channel"""
        while self.running:
            try:
                # Check if we have messages to republish
                if not self.message_queue.empty():
                    messages_to_publish = []
                    
                    # Collect all available messages
                    while not self.message_queue.empty() and len(messages_to_publish) < 10:
                        try:
                            message = self.message_queue.get_nowait()
                            messages_to_publish.append(message)
                        except queue.Empty:
                            break
                    
                    # Republish messages to dashboard
                    for message_data in messages_to_publish:
                        try:
                            await self.dashboard_channel.publish('telemetry_update', message_data)
                            self.stats["messages_republished"] += 1
                            
                            # Log source type
                            source_type = message_data.get('data_source', 'UNKNOWN')
                            logger.debug(f"📤 Republished {source_type} message {message_data.get('message_id', 'unknown')} to dashboard")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to republish message: {e}")
                            self.stats["errors"] += 1
                            self.stats["last_error"] = f"Republish error: {e}"
                            
                            # Put message back in queue for retry
                            self.message_queue.put(message_data)
                            break
                    
                    if messages_to_publish:
                        source_info = "MOCK" if self.mock_mode else "ESP32"
                        logger.info(f"📡 Republished {len(messages_to_publish)} {source_info} messages to dashboard")
                
                # Wait before next iteration
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in republish loop: {e}")
                self.stats["errors"] += 1
                self.stats["last_error"] = f"Republish loop error: {e}"
                await asyncio.sleep(1)
    
    async def print_stats(self):
        """Periodically print statistics"""
        while self.running:
            await asyncio.sleep(30)  # Print stats every 30 seconds
            
            mode_info = "MOCK DATA" if self.mock_mode else "ESP32 DATA"
            logger.info(f"📊 STATS ({mode_info}) - Received: {self.stats['messages_received']}, "
                       f"Republished: {self.stats['messages_republished']}, "
                       f"Errors: {self.stats['errors']}, "
                       f"Queue: {self.message_queue.qsize()}")
            
            if self.stats["last_error"]:
                logger.warning(f"⚠️ Last error: {self.stats['last_error']}")
    
    async def run(self):
        """Main execution loop"""
        mode_desc = "Mock Data Generator" if self.mock_mode else "ESP32 MQTT to Dashboard"
        logger.info(f"🚀 Starting Telemetry Bridge - {mode_desc}")
        
        # Connect to ESP32 data source (only if not in mock mode)
        if not await self.connect_esp32_subscriber():
            if not self.mock_mode:
                logger.error("Failed to connect to ESP32 data source")
                return
        
        # Connect to Dashboard output
        if not await self.connect_dashboard_publisher():
            logger.error("Failed to connect to Dashboard output")
            return
        
        self.running = True
        
        if self.mock_mode:
            logger.info("🎭 Bridge is active - generating MOCK telemetry data for dashboard")
        else:
            logger.info("🔄 Bridge is active - forwarding ESP32 telemetry to dashboard")
        
        try:
            # Run the appropriate loops based on mode
            if self.mock_mode:
                await asyncio.gather(
                    self.generate_mock_data_loop(),
                    self.republish_messages(),
                    self.print_stats()
                )
            else:
                await asyncio.gather(
                    self.republish_messages(),
                    self.print_stats()
                )
            
        except Exception as e:
            logger.error(f"💥 Unexpected error in main loop: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        
        if self.esp32_client:
            try:
                await self.esp32_client.close()
                logger.info("ESP32 Ably connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing ESP32 Ably connection: {e}")
        
        if self.dashboard_client:
            try:
                await self.dashboard_client.close()
                logger.info("Dashboard Ably connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Dashboard Ably connection: {e}")
        
        mode_info = "MOCK" if self.mock_mode else "REAL"
        logger.info(f"📊 Final stats ({mode_info}): {self.stats['messages_received']} received, "
                   f"{self.stats['messages_republished']} republished")

def get_user_preferences():
    """Get user preferences through terminal input"""
    print("\n" + "="*60)
    print("🚀 TELEMETRY BRIDGE - ESP32 to Dashboard")
    print("="*60)
    print()
    print("Choose data source mode:")
    print("1. 🔗 REAL DATA - Connect to ESP32 hardware")
    print("2. 🎭 MOCK DATA - Generate simulated telemetry data")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == '1':
                print("\n✅ Selected: REAL DATA mode")
                print("📡 Will attempt to connect to ESP32 hardware...")
                return False, 2.0  # mock_mode=False, default_interval
            elif choice == '2':
                print("\n✅ Selected: MOCK DATA mode")
                print("🎭 Will generate simulated telemetry data...")
                
                # Ask for mock data interval
                while True:
                    try:
                        interval_input = input("Enter mock data interval in seconds (default: 2.0): ").strip()
                        if interval_input == '':
                            interval = 2.0
                        else:
                            interval = float(interval_input)
                            if interval <= 0:
                                print("❌ Interval must be positive. Please try again.")
                                continue
                        print(f"📊 Mock data will be generated every {interval} seconds")
                        return True, interval  # mock_mode=True, custom_interval
                    except ValueError:
                        print("❌ Invalid number. Please enter a valid decimal number.")
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Operation cancelled by user.")
            sys.exit(0)
        except EOFError:
            print("\n\n🛑 Input ended unexpectedly.")
            sys.exit(0)

async def main():
    """Main entry point"""
    # Get user preferences
    mock_mode, mock_interval = get_user_preferences()
    
    # Update mock data interval
    global MOCK_DATA_INTERVAL
    MOCK_DATA_INTERVAL = mock_interval
    
    print("\n" + "-"*60)
    print("🔧 STARTING TELEMETRY BRIDGE...")
    print("-"*60)
    
    # Create bridge with appropriate mode
    bridge = TelemetryBridge(mock_mode=mock_mode)
    
    try:
        await bridge.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bridge stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
    
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bridge interrupted")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
