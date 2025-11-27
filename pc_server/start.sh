#!/bin/bash
# Start Rovy PC Server
# Runs AI models for Raspberry Pi client

cd "$(dirname "$0")"

echo "================================"
echo "  ROVY PC SERVER"
echo "================================"
echo "Listening: ws://0.0.0.0:8765"
echo "Tailscale: ws://100.121.110.125:8765"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Check dependencies
python3 -c "import websockets" 2>/dev/null || {
    echo "Installing websockets..."
    pip3 install websockets
}

python3 -c "import llama_cpp" 2>/dev/null || {
    echo "WARNING: llama-cpp-python not installed"
    echo "Install: pip install llama-cpp-python"
    echo "For GPU: CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python"
}

# Start server
python3 main.py

