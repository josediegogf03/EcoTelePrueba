import asyncio
import json
import logging
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemetryBridge:
    """
    Bridge class that subscribes to ESP32 MQTT telemetry data via Ably 
    and republishes it to the dashboard channel
    """
    
    def __init__(self):
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
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def connect_esp32_subscriber(self) -> bool:
        """Connect to ESP32 Ably channel to receive MQTT data"""
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
                            
                            logger.debug(f"📤 Republished message {message_data.get('message_id', 'unknown')} to dashboard")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to republish message: {e}")
                            self.stats["errors"] += 1
                            self.stats["last_error"] = f"Republish error: {e}"
                            
                            # Put message back in queue for retry
                            self.message_queue.put(message_data)
                            break
                    
                    if messages_to_publish:
                        logger.info(f"📡 Republished {len(messages_to_publish)} messages to dashboard")
                
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
            
            logger.info(f"📊 STATS - Received: {self.stats['messages_received']}, "
                       f"Republished: {self.stats['messages_republished']}, "
                       f"Errors: {self.stats['errors']}, "
                       f"Queue: {self.message_queue.qsize()}")
            
            if self.stats["last_error"]:
                logger.warning(f"⚠️ Last error: {self.stats['last_error']}")
    
    async def run(self):
        """Main execution loop"""
        logger.info("🚀 Starting Telemetry Bridge - ESP32 MQTT to Dashboard")
        
        # Connect to ESP32 data source
        if not await self.connect_esp32_subscriber():
            logger.error("Failed to connect to ESP32 data source")
            return
        
        # Connect to Dashboard output
        if not await self.connect_dashboard_publisher():
            logger.error("Failed to connect to Dashboard output")
            return
        
        self.running = True
        
        logger.info("🔄 Bridge is active - forwarding ESP32 telemetry to dashboard")
        
        try:
            # Run the republish loop and stats printer concurrently
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
        
        logger.info(f"📊 Final stats: {self.stats['messages_received']} received, "
                   f"{self.stats['messages_republished']} republished")

async def main():
    """Main entry point"""
    bridge = TelemetryBridge()
    
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
