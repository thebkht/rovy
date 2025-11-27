"""
Raspberry Pi Client Configuration
Connects to PC server via Tailscale
"""

# ===========================================
# TAILSCALE IPS - UPDATE IF NEEDED
# ===========================================
PC_SERVER_IP = "100.121.110.125"      # Your PC (cloud server)
RASPBERRY_PI_IP = "100.72.107.106"    # This Raspberry Pi
WEBSOCKET_PORT = 8765

# WebSocket URL to connect to
SERVER_URL = f"ws://{PC_SERVER_IP}:{WEBSOCKET_PORT}"

# ===========================================
# ROVER HARDWARE
# ===========================================
ROVER_SERIAL_PORT = "/dev/ttyACM0"
ROVER_BAUDRATE = 115200

# ===========================================
# CAMERA
# ===========================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
JPEG_QUALITY = 70  # Lower = smaller packets, faster streaming

# ===========================================
# AUDIO (ReSpeaker)
# ===========================================
SAMPLE_RATE = 16000
CHANNELS = 1  # Mono for speech recognition
CHUNK_SIZE = 1024
AUDIO_BUFFER_SECONDS = 3.0  # How much audio to buffer before sending

# ===========================================
# WAKE WORD
# ===========================================
WAKE_WORDS = ["hey rovy", "rovy", "hey robot"]

# ===========================================
# RECONNECTION
# ===========================================
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = -1  # -1 = infinite

