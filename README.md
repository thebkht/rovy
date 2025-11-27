# Rovy - AI Robot Assistant

Cloud-based robot assistant using Raspberry Pi + PC via Tailscale.

## Architecture

```
┌─────────────────────┐         Tailscale          ┌─────────────────────┐
│   RASPBERRY PI      │◄─────────────────────────►│        PC           │
│   100.72.107.106    │       WebSocket            │   100.121.110.125   │
│                     │                            │                     │
│ • Rover control     │  ──── Audio/Video ────►   │ • LLM (Gemma/Llama) │
│ • Camera            │                            │ • Vision (LLaVA)    │
│ • Microphone        │  ◄─── Commands ────────   │ • Whisper STT       │
│ • Speaker           │       (speak, move, etc)   │ • TTS               │
└─────────────────────┘                            └─────────────────────┘
```

## Folder Structure

```
rovy/
├── raspberry_client/    # 👈 Runs on Raspberry Pi (this device)
│   ├── main.py          # Main client
│   ├── rover.py         # Rover serial control
│   ├── config.py        # Network config (Tailscale IPs)
│   └── start.sh
│
├── pc_server/           # 👈 Copy to your PC
│   ├── main.py          # WebSocket server
│   ├── assistant.py     # LLM inference
│   ├── speech.py        # STT/TTS
│   └── start.sh
│
├── esp32_firmware/      # ESP32 rover firmware
│
└── jetson_legacy/       # Old Jetson code (archived)
```

## Quick Start

### 1. On PC (100.121.110.125)
```bash
# Copy pc_server to your PC
scp -r pc_server/ user@100.121.110.125:~/rovy_server/

# SSH to PC and run
cd ~/rovy_server
pip install -r requirements.txt
./start.sh
```

### 2. On Raspberry Pi (this device)
```bash
cd ~/rovy_client/raspberry_client
pip install -r requirements.txt
./start.sh
```

## Network (Tailscale)

| Device | Tailscale IP | Role |
|--------|--------------|------|
| PC | 100.121.110.125 | Server (AI processing) |
| Raspberry Pi | 100.72.107.106 | Client (rover control) |

WebSocket: `ws://100.121.110.125:8765`
