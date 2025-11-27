#!/usr/bin/env python3
"""
Rovy PC Server
Runs AI models and processes requests from Raspberry Pi client.

Usage:
    python main.py
"""
import asyncio
import json
import time
import base64
import signal
import sys
import logging
from datetime import datetime
from typing import Set, Optional

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('Server')

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# Import AI modules
from assistant import CloudAssistant
from speech import SpeechProcessor


class RovyServer:
    """
    Server that runs on PC.
    - Receives audio/video from Raspberry Pi
    - Runs AI models (LLM, Vision, STT)
    - Sends commands back to Raspberry Pi
    """
    
    def __init__(self):
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        
        # Last received data
        self.last_image: Optional[bytes] = None
        self.last_sensors = {}
        
        logger.info("=" * 50)
        logger.info("  ROVY PC SERVER")
        logger.info(f"  Listening on: ws://0.0.0.0:{config.PORT}")
        logger.info(f"  Tailscale: ws://{config.PC_SERVER_IP}:{config.PORT}")
        logger.info("=" * 50)
        
        # Initialize AI components
        logger.info("Loading AI models...")
        
        self.assistant = CloudAssistant(
            text_model_path=config.TEXT_MODEL_PATH,
            vision_model_path=config.VISION_MODEL_PATH,
            vision_mmproj_path=config.VISION_MMPROJ_PATH,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.N_CTX,
            lazy_load_vision=True
        )
        
        self.speech = SpeechProcessor(
            whisper_model=config.WHISPER_MODEL,
            tts_engine=config.TTS_ENGINE
        )
        
        logger.info("✅ Server ready!")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str = "/"):
        """Handle new client connection."""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔗 Client connected: {client_addr}")
        
        self.clients.add(websocket)
        
        try:
            # Send welcome
            await self.send_speak(websocket, f"Connected to {config.ASSISTANT_NAME}")
            
            async for message in websocket:
                try:
                    await self.handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Message error: {e}")
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_addr}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket: WebSocketServerProtocol, raw_message: str):
        """Process incoming message from client."""
        msg = json.loads(raw_message)
        msg_type = msg.get('type', '')
        
        if msg_type == 'audio_data':
            await self.handle_audio(websocket, msg)
        
        elif msg_type == 'image_data':
            await self.handle_image(websocket, msg)
        
        elif msg_type == 'text_query':
            await self.handle_text_query(websocket, msg)
        
        elif msg_type == 'sensor_data':
            self.handle_sensor_data(msg)
        
        elif msg_type == 'wake_word':
            await self.send_speak(websocket, "Yes? I'm listening.")
        
        elif msg_type == 'ping':
            await websocket.send(json.dumps({"type": "pong"}))
    
    async def handle_audio(self, websocket: WebSocketServerProtocol, msg: dict):
        """Process audio - run speech recognition."""
        logger.info("🎤 Processing audio...")
        
        try:
            audio_bytes = base64.b64decode(msg.get('audio_base64', ''))
            sample_rate = msg.get('sample_rate', 16000)
            
            # Run Whisper
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.speech.transcribe, audio_bytes, sample_rate
            )
            
            if text:
                logger.info(f"📝 Heard: '{text}'")
                
                # Check for wake word
                text_lower = text.lower()
                wake_detected = any(w in text_lower for w in config.WAKE_WORDS)
                
                if wake_detected:
                    # Remove wake word
                    query = text_lower
                    for wake in config.WAKE_WORDS:
                        query = query.replace(wake, "").strip()
                    
                    if query:
                        await self.process_query(websocket, query)
                    else:
                        await self.send_speak(websocket, "Yes? How can I help?")
                else:
                    # Process as query anyway
                    await self.process_query(websocket, text)
            
        except Exception as e:
            logger.error(f"Audio error: {e}")
    
    async def handle_image(self, websocket: WebSocketServerProtocol, msg: dict):
        """Store latest image from camera."""
        try:
            self.last_image = base64.b64decode(msg.get('image_base64', ''))
            # Just store it - vision queries will use it
        except Exception as e:
            logger.error(f"Image error: {e}")
    
    async def handle_text_query(self, websocket: WebSocketServerProtocol, msg: dict):
        """Handle direct text query."""
        text = msg.get('text', '')
        use_vision = msg.get('include_vision', False)
        
        logger.info(f"💬 Query: '{text}'")
        
        await self.process_query(websocket, text, use_vision=use_vision)
    
    def handle_sensor_data(self, msg: dict):
        """Store sensor readings."""
        self.last_sensors = {
            'battery_voltage': msg.get('battery_voltage'),
            'battery_percent': msg.get('battery_percent'),
            'temperature': msg.get('temperature'),
            'imu': {
                'roll': msg.get('imu_roll'),
                'pitch': msg.get('imu_pitch'),
                'yaw': msg.get('imu_yaw')
            }
        }
    
    async def process_query(self, websocket: WebSocketServerProtocol, query: str, use_vision: bool = None):
        """Process a query and send response."""
        logger.info(f"🎯 Processing: '{query}'")
        
        # Auto-detect if vision is needed
        if use_vision is None:
            vision_keywords = ['see', 'look', 'what is', 'who is', 'describe', 'show', 'camera', 'front']
            use_vision = any(kw in query.lower() for kw in vision_keywords)
        
        try:
            if use_vision and self.last_image:
                # Use vision model
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.assistant.ask_with_vision, query, self.last_image
                )
            else:
                # Use text model
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.assistant.ask, query
                )
            
            await self.send_speak(websocket, response)
            
            # Check for movement commands
            movement = self.assistant.extract_movement(response, query)
            if movement:
                await self.send_move(websocket, **movement)
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            await self.send_speak(websocket, "Sorry, I had trouble with that.")
    
    # === Send commands to client ===
    
    async def send_speak(self, websocket: WebSocketServerProtocol, text: str):
        """Send TTS response to client."""
        # Generate audio
        audio_b64 = None
        try:
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self.speech.synthesize, text
            )
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
        
        msg = {
            "type": "speak",
            "text": text,
            "audio_base64": audio_b64,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send(json.dumps(msg))
        logger.info(f"🔊 Sent: '{text[:50]}...'")
    
    async def send_move(self, websocket: WebSocketServerProtocol, 
                        direction: str, distance: float = 0.5, speed: str = "medium"):
        """Send movement command."""
        msg = {
            "type": "move",
            "direction": direction,
            "distance": distance,
            "speed": speed,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send(json.dumps(msg))
        logger.info(f"🚗 Move: {direction} {distance}m at {speed}")
    
    async def send_gimbal(self, websocket: WebSocketServerProtocol,
                          pan: float, tilt: float, action: str = "move"):
        """Send gimbal command."""
        msg = {
            "type": "gimbal",
            "pan": pan,
            "tilt": tilt,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send(json.dumps(msg))
    
    async def send_display(self, websocket: WebSocketServerProtocol, lines: list):
        """Send display command."""
        msg = {
            "type": "display",
            "lines": lines[:4],
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send(json.dumps(msg))
    
    async def send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send error message."""
        msg = {
            "type": "error",
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send(json.dumps(msg))
    
    async def start(self):
        """Start the WebSocket server."""
        self.running = True
        
        async with serve(
            self.handle_connection,
            config.HOST,
            config.PORT,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info(f"✅ Server running on ws://{config.HOST}:{config.PORT}")
            logger.info("Waiting for Raspberry Pi client...")
            
            while self.running:
                await asyncio.sleep(1)
    
    def stop(self):
        """Stop the server."""
        self.running = False
        logger.info("🛑 Server stopping...")


server: RovyServer = None


def signal_handler(sig, frame):
    logger.info("\n👋 Shutting down...")
    if server:
        server.stop()
    sys.exit(0)


async def main():
    global server
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    ROVY PC SERVER                         ║
    ║              AI-Powered Robot Assistant                   ║
    ║                                                           ║
    ║  Using LOCAL models:                                      ║
    ║  • Text: Gemma/Llama via llama.cpp                       ║
    ║  • Vision: LLaVA via llama.cpp                           ║
    ║  • Speech: Whisper (STT) + espeak/Piper (TTS)            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = RovyServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

