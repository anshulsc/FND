# src/modules/embedding_utils.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Model Loading (Singleton Pattern) ---
# This ensures we only load the power-hungry model once.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None

def _get_clip_model():
    """Loads and caches the CLIP model and processor."""
    global _model, _processor
    if _model is None:
        print("INFO: Loading CLIP model for the first time...")
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("INFO: CLIP model loaded successfully.")
    return _model, _processor

# --- Embedding Functions ---

def get_image_embedding(image_path: str) -> list[float]:
    """Computes the embedding for a single image file."""
    model, processor = _get_clip_model()
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) # Normalize
        
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"ERROR: Could not process image {image_path}. Reason: {e}")
        return None

def get_text_embedding(text: str) -> list[float]:
    """Computes the embedding for a single text string."""
    model, processor = _get_clip_model()
    try:
        with torch.no_grad():
            inputs = processor(text=text, return_tensors="pt", padding=True).to(DEVICE)
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) # Normalize

        return text_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"ERROR: Could not process text. Reason: {e}")
        return None