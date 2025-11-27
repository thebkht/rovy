# Rovy PC Server

This runs on your **PC** (100.121.110.125) and processes AI tasks for the Raspberry Pi client.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. For GPU acceleration (NVIDIA):
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall

# 3. Download models (see below)

# 4. Start server
python main.py
```

## Download Models

### Text Model (choose one)
```bash
# Gemma 2B (small, fast) - RECOMMENDED
wget https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf -P ~/.cache/

# Or Llama 2 7B (larger, better quality)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -P ~/.cache/
```

### Vision Model (optional - for "what do you see?" queries)
```bash
# LLaVA v1.5 7B
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf -P ~/.cache/
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf -P ~/.cache/
```

### Set Model Paths (optional)
```bash
export ROVY_TEXT_MODEL=/path/to/model.gguf
export ROVY_VISION_MODEL=/path/to/llava.gguf
export ROVY_VISION_MMPROJ=/path/to/mmproj.gguf
```

## Network

The server listens on:
- `ws://0.0.0.0:8765` (all interfaces)
- Via Tailscale: `ws://100.121.110.125:8765`

The Raspberry Pi connects via Tailscale at `100.72.107.106`.

## Configuration

Edit `config.py` to change:
- Port number
- Model paths
- Whisper model size (tiny/base/small/medium/large)
- TTS engine (espeak/piper)

## Troubleshooting

### "Model not found"
Set the path explicitly:
```bash
export ROVY_TEXT_MODEL=/full/path/to/model.gguf
```

### "CUDA out of memory"
Reduce GPU layers in `config.py`:
```python
N_GPU_LAYERS = 20  # Instead of -1
```

### Slow speech recognition
Use smaller Whisper model:
```python
WHISPER_MODEL = "tiny"  # Fastest
```

