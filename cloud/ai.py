"""
Cloud AI Assistant - Using Qwen2-VL for Vision + Text
Fast VLM running on PC with RTX 4080 SUPER.
"""
import os
import re
import gc
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger('Assistant')

# Try to import Qwen2-VL dependencies
QWEN_VL_OK = False
try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_VL_OK = True
except ImportError as e:
    logger.warning(f"Qwen2-VL not available: {e}")
    logger.warning("Install: pip install transformers accelerate qwen-vl-utils torch")

try:
    from PIL import Image
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


class CloudAssistant:
    """AI Assistant using Qwen2-VL for both vision and text."""
    
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    
    def __init__(
        self,
        model_id: str = None,
        max_pixels: int = 1280 * 720,  # Limit image size for speed
        lazy_load: bool = False
    ):
        self.model_id = model_id or self.MODEL_ID
        self.max_pixels = max_pixels
        
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not QWEN_VL_OK:
            logger.error("Qwen2-VL dependencies not installed!")
            return
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_id}")
        
        if not lazy_load:
            self._load_model()
    
    def _free_gpu(self):
        """Free GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_model(self):
        """Load the Qwen2-VL model."""
        if self.model is not None:
            return
        
        logger.info("Loading Qwen2-VL model...")
        start = time.time()
        
        self._free_gpu()
        
        try:
            # Load model fully on GPU
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation="eager",  # or "flash_attention_2" if installed
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=256 * 256,
                max_pixels=self.max_pixels,
            )
            
            logger.info(f"✅ Qwen2-VL loaded in {time.time() - start:.1f}s")
            
            # Log VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"VRAM used: {vram_used:.1f}GB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.processor = None
    
    def ask(self, question: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """Ask a text-only question."""
        if not self.model:
            self._load_model()
            if not self.model:
                return "Model not available."
        
        logger.info(f"Query: {question}")
        start = time.time()
        
        try:
            # Add time context if needed
            time_ctx = ""
            if any(p in question.lower() for p in ['time', 'date', 'today', 'day']):
                time_ctx = f"Current: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}. "
            
            # Qwen2-VL works better with instruction in user message
            system_instruction = "You are Jarvis, a friendly robot assistant. Always refer to yourself as Jarvis. Be concise (under 50 words)."
            prompt = f"{system_instruction}\n{time_ctx}{question}"
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Prepare input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=3,  # Prevent cutting off too early
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # Decode response
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            answer = self._clean(answer)
            logger.info(f"Response in {time.time() - start:.1f}s")
            return answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error: {e}"
    
    def ask_with_vision(self, question: str, image, max_tokens: int = 200) -> str:
        """Ask a question about an image."""
        if not self.model:
            self._load_model()
            if not self.model:
                return "Model not available."
        
        logger.info(f"Vision query: {question}")
        start = time.time()
        
        try:
            # Convert to PIL Image
            pil_img = self._to_pil(image)
            if not pil_img:
                return "Could not process image."
            
            # Resize if too large (for speed)
            pil_img = self._resize_image(pil_img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Prepare input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=True,
                )
            
            # Decode response
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            answer = self._clean(answer)
            logger.info(f"Vision response in {time.time() - start:.1f}s")
            return answer if answer else "I couldn't understand what I'm seeing."
            
        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            return f"Error: {e}"
    
    def _resize_image(self, img: Image.Image, max_size: int = 1280) -> Image.Image:
        """Resize image for faster processing."""
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img
    
    def _to_pil(self, image) -> Optional[Image.Image]:
        """Convert various formats to PIL Image."""
        if not PIL_OK:
            return None
        
        try:
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image)).convert("RGB")
            if isinstance(image, np.ndarray):
                if CV2_OK and len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image).convert("RGB")
            if isinstance(image, str) and os.path.exists(image):
                return Image.open(image).convert("RGB")
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
        return None
    
    def _clean(self, text: str) -> str:
        """Clean model output for TTS."""
        text = text.replace('</s>', '').replace('#', '')
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Remove emojis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_movement(self, response: str, query: str) -> Optional[Dict[str, Any]]:
        """Extract movement commands from text."""
        text = f"{query} {response}".lower()
        
        patterns = {
            'forward': [r'go\s+forward', r'move\s+forward', r'ahead'],
            'backward': [r'go\s+back', r'move\s+back', r'reverse'],
            'left': [r'turn\s+left', r'go\s+left'],
            'right': [r'turn\s+right', r'go\s+right'],
            'stop': [r'\bstop\b', r'halt']
        }
        
        for direction, pats in patterns.items():
            for pat in pats:
                if re.search(pat, text):
                    dist = 0.5
                    if 'little' in text:
                        dist = 0.2
                    elif 'far' in text or 'lot' in text:
                        dist = 1.0
                    
                    speed = 'medium'
                    if 'slow' in text:
                        speed = 'slow'
                    elif 'fast' in text:
                        speed = 'fast'
                    
                    return {'direction': direction, 'distance': dist, 'speed': speed}
        
        return None
