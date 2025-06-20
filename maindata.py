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

try:
    from ably import AblyRealtime
except ImportError:
    print("Error: Ably library not installed. Run: pip install ably")
    sys.exit(1)

# Configuration
ABLY_API_KEY = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
CHANNEL_NAME = "telemetry-dashboard-channel"
PUBLISH_INTERVAL = 2.0  # seconds

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemetryPublisher:
    def __init__(self):
        self.ably = None
        self.channel = None
        self.running = False
        self.message_count = 0
        
        # Simulation state
        self.cumulative_distance = 0.0
        self.cumulative_energy = 0.0
        self.simulation_time = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def connect(self) -> bool:
        """Establish connection to Ably"""
        try:
            logger.info("Connecting to Ably...")
            
            # Create Ably Realtime client
            self.ably = AblyRealtime(ABLY_API_KEY)
            
            # Get channel
            self.channel = self.ably.channels.get(CHANNEL_NAME)
            
            # Wait for connection
            await self._wait_for_connection()
            
            logger.info(f"✅ Connected to Ably channel: {CHANNEL_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ably: {e}")
            return False
    
    async def _wait_for_connection(self, timeout=10):
        """Wait for Ably connection to be established"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                connection_state = self.ably.connection.state
                if connection_state == 'connected':
                    return
                await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(0.1)
        
        # If we can't check state, assume connected after timeout
        logger.info("Connection state check timeout, assuming connected")
    
    def generate_telemetry_data(self) -> Dict[str, Any]:
        """Generate realistic mock telemetry data"""
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
        energy_delta = power * PUBLISH_INTERVAL
        distance_delta = speed * PUBLISH_INTERVAL
        
        self.cumulative_energy += energy_delta
        self.cumulative_distance += distance_delta
        
        # Generate GPS coordinates (simulated route)
        base_lat, base_lon = 40.7128, -74.0060
        lat_offset = 0.001 * math.sin(self.simulation_time * 0.05)
        lon_offset = 0.001 * math.cos(self.simulation_time * 0.05)
        
        latitude = base_lat + lat_offset + random.gauss(0, 0.0001)
        longitude = base_lon + lon_offset + random.gauss(0, 0.0001)
        
        self.simulation_time += 1
        
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
            'message_id': self.message_count + 1,
            'uptime_seconds': self.simulation_time * PUBLISH_INTERVAL
        }
    
    async def publish_data(self, data: Dict[str, Any]) -> bool:
        """Publish telemetry data to Ably channel"""
        try:
            # Publish message
            await self.channel.publish('telemetry_update', data)
            
            self.message_count += 1
            
            # Log every 10th message
            if self.message_count % 10 == 0:
                logger.info(f"📡 Published {self.message_count} messages - "
                           f"Speed: {data['speed_ms']} m/s, Power: {data['power_w']:.1f} W")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish data: {e}")
            return False
    
    async def run(self):
        """Main publishing loop"""
        logger.info("🚀 Starting Telemetry Publisher")
        
        # Connect to Ably
        if not await self.connect():
            logger.error("Failed to connect to Ably")
            return
        
        self.running = True
        consecutive_failures = 0
        
        try:
            while self.running:
                # Generate and publish data
                data = self.generate_telemetry_data()
                
                if await self.publish_data(data):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    
                    # If too many failures, try to reconnect
                    if consecutive_failures >= 5:
                        logger.warning("Too many failures, attempting to reconnect...")
                        if await self.connect():
                            consecutive_failures = 0
                        else:
                            logger.error("Reconnection failed")
                            break
                
                # Wait for next publish cycle
                await asyncio.sleep(PUBLISH_INTERVAL)
                
        except Exception as e:
            logger.error(f"💥 Unexpected error in main loop: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        
        if self.ably:
            try:
                await self.ably.close()
                logger.info("Ably connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Ably connection: {e}")
        
        logger.info(f"📊 Final stats: {self.message_count} messages published")

async def main():
    """Main entry point"""
    publisher = TelemetryPublisher()
    
    try:
        await publisher.run()
    except KeyboardInterrupt:
        logger.info("🛑 Publisher stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
    
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Publisher interrupted")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

