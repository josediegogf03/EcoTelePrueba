import asyncio
import os
import random
import time
from datetime import datetime, timedelta
import logging
import json
import signal
import sys
from typing import Dict, Any

try:
    from ably import AblyRealtime
    from ably.types.options import Options
    ABLY_AVAILABLE = True
except ImportError:
    ABLY_AVAILABLE = False
    logging.error("Ably library not installed. Install with: pip install ably")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('telemetry_publisher.log', mode='a')
    ]
)

# --- Configuration ---
ABLY_API_KEY_FALLBACK = "DxuYSw.fQHpug:sa4tOcqWDkYBW9ht56s7fT0G091R1fyXQc6mc8WthxQ"
ABLY_API_KEY = os.environ.get('ABLY_API_KEY', ABLY_API_KEY_FALLBACK)
TELEMETRY_CHANNEL_NAME = "telemetry-dashboard-channel"

# Publisher settings
PUBLISH_INTERVAL = 2.0  # seconds between data points
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_DELAY_BASE = 5  # seconds
MAX_RECONNECT_DELAY = 60  # seconds

# Simulation parameters
SIMULATION_CONFIG = {
    'speed': {'base': 15.0, 'variation': 3.0, 'min': 0.0, 'max': 50.0},
    'voltage': {'base': 48.0, 'variation': 2.0, 'min': 40.0, 'max': 55.0},
    'current': {'base': 8.0, 'variation': 2.0, 'min': 0.0, 'max': 20.0},
    'location': {
        'lat_base': 40.7128, 'lon_base': -74.0060,
        'drift_rate': 0.0001, 'noise': 0.0005
    }
}

class TelemetryPublisher:
    def __init__(self):
        self.realtime = None
        self.channel = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.last_publish_time = None
        self.total_messages_sent = 0
        self.start_time = datetime.now()
        
        # Simulation state
        self.cumulative_distance = 0.0
        self.cumulative_energy = 0.0
        self.simulation_time_offset = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    def _validate_api_key(self) -> bool:
        """Validate Ably API key format and availability"""
        if not ABLY_API_KEY:
            logging.error("ABLY_API_KEY is empty or None")
            return False
        
        if len(ABLY_API_KEY) < 20:  # Basic length check
            logging.error("ABLY_API_KEY appears to be too short")
            return False
        
        if "YOUR_ACTUAL_API_KEY_PLACEHOLDER" in ABLY_API_KEY:
            logging.error("ABLY_API_KEY contains placeholder text")
            return False
        
        # Log which key source we're using (masked for security)
        if ABLY_API_KEY == ABLY_API_KEY_FALLBACK:
            logging.info(f"Using fallback API key: {ABLY_API_KEY[:8]}...")
        else:
            logging.info(f"Using environment API key: {ABLY_API_KEY[:8]}...")
        
        return True
    
    async def _connect_to_ably(self) -> bool:
        """Establish connection to Ably with retry logic"""
        if not self._validate_api_key():
            return False
        
        try:
            logging.info("Connecting to Ably...")
            options = Options(
                key=ABLY_API_KEY,
                log_level=logging.WARNING,
                auto_connect=True,
                disconnect_on_suspend=False,
                fallback_hosts_use_default=True
            )
            
            self.realtime = AblyRealtime(options)
            
            # Set up connection state listeners
            self.realtime.connection.on('connected', self._on_connected)
            self.realtime.connection.on('disconnected', self._on_disconnected)
            self.realtime.connection.on('suspended', self._on_suspended)
            self.realtime.connection.on('failed', self._on_failed)
            
            # Wait for connection with timeout
            await asyncio.wait_for(
                self.realtime.connection.once_async('connected'), 
                timeout=15.0
            )
            
            # Get channel
            self.channel = self.realtime.channels.get(TELEMETRY_CHANNEL_NAME)
            
            logging.info(f"Successfully connected to Ably and joined channel: {TELEMETRY_CHANNEL_NAME}")
            self.reconnect_attempts = 0
            return True
            
        except asyncio.TimeoutError:
            logging.error("Connection timeout - Ably took too long to connect")
            return False
        except Exception as e:
            logging.error(f"Failed to connect to Ably: {e}")
            return False
    
    def _on_connected(self, state_change):
        """Handle successful connection"""
        logging.info("Ably connection established")
    
    def _on_disconnected(self, state_change):
        """Handle disconnection"""
        logging.warning(f"Ably disconnected: {state_change.reason}")
    
    def _on_suspended(self, state_change):
        """Handle connection suspension"""
        logging.warning(f"Ably connection suspended: {state_change.reason}")
    
    def _on_failed(self, state_change):
        """Handle connection failure"""
        logging.error(f"Ably connection failed: {state_change.reason}")
    
    def _calculate_reconnect_delay(self) -> float:
        """Calculate exponential backoff delay"""
        delay = min(
            RECONNECT_DELAY_BASE * (2 ** self.reconnect_attempts),
            MAX_RECONNECT_DELAY
        )
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter
    
    async def _reconnect_with_backoff(self) -> bool:
        """Attempt to reconnect with exponential backoff"""
        while self.reconnect_attempts < MAX_RECONNECT_ATTEMPTS and self.is_running:
            self.reconnect_attempts += 1
            delay = self._calculate_reconnect_delay()
            
            logging.info(f"Reconnection attempt {self.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} in {delay:.1f}s")
            await asyncio.sleep(delay)
            
            if not self.is_running:
                break
            
            if await self._connect_to_ably():
                return True
        
        logging.error(f"Failed to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts")
        return False
    
    def _generate_telemetry_data(self) -> Dict[str, Any]:
        """Generate realistic telemetry data with proper state tracking"""
        current_time = datetime.now()
        
        # Calculate time delta for realistic accumulation
        time_delta = PUBLISH_INTERVAL
        
        # Generate speed with realistic variation
        speed_config = SIMULATION_CONFIG['speed']
        base_speed = speed_config['base']
        
        # Add some periodic variation (like hills, traffic, etc.)
        periodic_factor = 2.0 * (1 + 0.3 * random.sin(self.simulation_time_offset * 0.1))
        speed_variation = random.gauss(base_speed, speed_config['variation'])
        speed = max(speed_config['min'], 
                   min(speed_config['max'], speed_variation * periodic_factor))
        
        # Generate voltage (battery voltage decreases over time slightly)
        voltage_config = SIMULATION_CONFIG['voltage']
        voltage_drift = -0.001 * self.simulation_time_offset  # Slow discharge
        voltage = max(voltage_config['min'],
                     min(voltage_config['max'],
                         voltage_config['base'] + voltage_drift + 
                         random.gauss(0, voltage_config['variation'])))
        
        # Generate current (higher when speed is higher)
        current_config = SIMULATION_CONFIG['current']
        speed_factor = (speed / base_speed) * 0.5 + 0.5  # 0.5 to 1.5 multiplier
        current = max(current_config['min'],
                     min(current_config['max'],
                         current_config['base'] * speed_factor + 
                         random.gauss(0, current_config['variation'])))
        
        # Calculate power and accumulate energy
        power = voltage * current
        energy_delta = power * time_delta
        self.cumulative_energy += energy_delta
        
        # Accumulate distance
        distance_delta = speed * time_delta
        self.cumulative_distance += distance_delta
        
        # Generate GPS coordinates with realistic drift
        loc_config = SIMULATION_CONFIG['location']
        drift_x = loc_config['drift_rate'] * self.simulation_time_offset
        drift_y = loc_config['drift_rate'] * self.simulation_time_offset * 0.7
        
        latitude = (loc_config['lat_base'] + drift_x + 
                   random.gauss(0, loc_config['noise']))
        longitude = (loc_config['lon_base'] + drift_y + 
                    random.gauss(0, loc_config['noise']))
        
        self.simulation_time_offset += 1
        
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
            'message_id': self.total_messages_sent + 1,
            'uptime_seconds': int((current_time - self.start_time).total_seconds())
        }
    
    async def _publish_data_point(self, data: Dict[str, Any]) -> bool:
        """Publish a single data point with error handling"""
        if not self.channel:
            logging.error("Channel not available for publishing")
            return False
        
        try:
            await asyncio.wait_for(
                self.channel.publish('telemetry_update', data),
                timeout=10.0
            )
            
            self.total_messages_sent += 1
            self.last_publish_time = datetime.now()
            
            # Log every 10th message to avoid spam
            if self.total_messages_sent % 10 == 0:
                logging.info(f"Published {self.total_messages_sent} messages "
                           f"(Speed: {data['speed_ms']} m/s, Power: {data['power_w']:.1f} W)")
            
            return True
            
        except asyncio.TimeoutError:
            logging.error("Publish timeout")
            return False
        except Exception as e:
            logging.error(f"Error publishing data: {e}")
            return False
    
    async def _health_check(self):
        """Periodic health check"""
        while self.is_running:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            if not self.is_running:
                break
            
            # Check if we've published recently
            if (self.last_publish_time and 
                (datetime.now() - self.last_publish_time).total_seconds() > 60):
                logging.warning("No successful publishes in the last minute")
            
            # Log statistics
            uptime = (datetime.now() - self.start_time).total_seconds()
            rate = self.total_messages_sent / uptime if uptime > 0 else 0
            logging.info(f"Health check - Messages: {self.total_messages_sent}, "
                        f"Rate: {rate:.2f}/s, Uptime: {uptime:.0f}s")
    
    async def run(self):
        """Main publishing loop"""
        logging.info("Starting Telemetry Publisher...")
        
        # Initial connection
        if not await self._connect_to_ably():
            logging.error("Failed to establish initial connection")
            return
        
        self.is_running = True
        
        # Start health check task
        health_task = asyncio.create_task(self._health_check())
        
        try:
            while self.is_running:
                # Check connection state
                if (not self.realtime or 
                    self.realtime.connection.state not in ['connected']):
                    logging.warning("Connection lost, attempting to reconnect...")
                    
                    if not await self._reconnect_with_backoff():
                        logging.error("Failed to reconnect, stopping publisher")
                        break
                
                # Generate and publish data
                try:
                    data_point = self._generate_telemetry_data()
                    
                    if await self._publish_data_point(data_point):
                        # Reset reconnect attempts on successful publish
                        self.reconnect_attempts = 0
                    else:
                        logging.warning("Failed to publish data point")
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                
                # Wait for next publish cycle
                await asyncio.sleep(PUBLISH_INTERVAL)
                
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
        finally:
            self.is_running = False
            health_task.cancel()
            await self._cleanup()
    
    async def _cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up resources...")
        
        if self.realtime:
            try:
                await asyncio.wait_for(self.realtime.close(), timeout=5.0)
                logging.info("Ably connection closed successfully")
            except Exception as e:
                logging.error(f"Error closing Ably connection: {e}")
        
        # Log final statistics
        uptime = (datetime.now() - self.start_time).total_seconds()
        rate = self.total_messages_sent / uptime if uptime > 0 else 0
        logging.info(f"Publisher stopped. Total messages: {self.total_messages_sent}, "
                    f"Average rate: {rate:.2f}/s, Total uptime: {uptime:.0f}s")

async def main():
    """Main entry point"""
    if not ABLY_AVAILABLE:
        logging.error("Ably library not available")
        return 1
    
    publisher = TelemetryPublisher()
    
    try:
        await publisher.run()
        return 0
    except KeyboardInterrupt:
        logging.info("Publisher stopped by user")
        return 0
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("maindata.py stopped by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
