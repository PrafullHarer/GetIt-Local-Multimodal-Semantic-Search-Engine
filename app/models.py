import os
import torch
import logging
import re
from typing import List
from functools import lru_cache
import torch.nn.functional as F  # FIX: required for safe normalization with new HF outputs

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
BLIP_CACHE_DIR = os.path.join(PROJECT_ROOT, "models", "blip")
CLIP_CACHE_DIR = os.path.join(PROJECT_ROOT, "models", "clip")

# Create model directories if they don't exist
os.makedirs(BLIP_CACHE_DIR, exist_ok=True)
os.makedirs(CLIP_CACHE_DIR, exist_ok=True)

# Model names
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def load_model_safe(model_class, model_name, cache_dir, **kwargs):
    """
    Try loading from local cache first. If failed, download from hub.
    """
    try:
        logger.info(f"Loading {model_name} from local cache...")
        return model_class.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            **kwargs
        )
    except Exception:
        logger.info(f"Local cache not found for {model_name}. Downloading...")
        return model_class.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            **kwargs
        )

# Load models once at module import
logger.info("Loading BLIP model...")
blip_processor = load_model_safe(BlipProcessor, BLIP_MODEL_NAME, BLIP_CACHE_DIR)
blip_model = load_model_safe(BlipForConditionalGeneration, BLIP_MODEL_NAME, BLIP_CACHE_DIR).to(device)

logger.info("Loading CLIP model...")
clip_processor = load_model_safe(CLIPProcessor, CLIP_MODEL_NAME, CLIP_CACHE_DIR)
clip_model = load_model_safe(CLIPModel, CLIP_MODEL_NAME, CLIP_CACHE_DIR).to(device)

logger.info("Models loaded successfully!")


def generate_caption(image, max_tokens=60, num_beams=4):
    """
    Generate caption for an image using BLIP.
    """
    try:
        inputs = blip_processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=num_beams
        )
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "Error generating caption"


def get_image_embedding(image):
    """
    Generate CLIP embedding for an image.
    """
    try:
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embeds = clip_model.get_image_features(**inputs)

        # FIX: Handle Transformers 5.0+ output types
        if not isinstance(embeds, torch.Tensor):
            if hasattr(embeds, 'image_embeds'):
                embeds = embeds.image_embeds
            elif hasattr(embeds, 'pooler_output'):
                embeds = embeds.pooler_output
            else:
                embeds = embeds[0]

        # FIX: .norm() on HF outputs breaks in newer transformers; normalize tensor safely
        embeds = F.normalize(embeds, dim=-1)

        return embeds[0].cpu().numpy()
    except Exception as e:
        logger.error(f"Error generating image embedding: {e}")
        return None


@lru_cache(maxsize=128)
def get_text_embedding(text):
    """
    Generate CLIP embedding for text with caching.
    """
    try:
        inputs = clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(device)
        with torch.no_grad():
            embeds = clip_model.get_text_features(**inputs)

        # FIX: Handle Transformers 5.0+ output types
        if not isinstance(embeds, torch.Tensor):
            if hasattr(embeds, 'text_embeds'):
                embeds = embeds.text_embeds
            elif hasattr(embeds, 'pooler_output'):
                embeds = embeds.pooler_output
            else:
                embeds = embeds[0]

        # FIX: same normalization issue as image embeddings
        embeds = F.normalize(embeds, dim=-1)

        return embeds[0].cpu().numpy()
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}")
        return None


def get_model_info():
    """Return information about loaded models."""
    return {
        "blip_model": BLIP_MODEL_NAME,
        "clip_model": CLIP_MODEL_NAME,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }


def caption_to_keywords(caption: str) -> List[str]:
    """
    Extract keywords from caption.
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9 ]+", " ", caption)
    words = caption.split()

    stopwords = {
        "a", "an", "the", "and", "or", "of", "in", "on", "with", "for", "to",
        "from", "by", "at", "near", "over", "under", "is", "are", "this",
        "that", "these", "those", "into", "onto", "as", "it", "its", "his",
        "her", "their", "our", "your", "you", "i", "we", "they", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can", "am", "was", "were"
    }

    filtered = [w for w in words if w not in stopwords and len(w) > 2]

    seen = set()
    keywords: List[str] = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            keywords.append(w)

    return keywords
