# models.py — AI Model Layer

> Loads and manages the BLIP and CLIP neural networks. Provides functions for captioning, embedding, and keyword extraction.

---

## Overview

This is the **core AI module**. It loads two neural networks at startup and exposes simple functions that the rest of the app calls. Everything ML-related flows through this file.

| Model | Identifier | Purpose |
|-------|-----------|---------|
| **BLIP** | `Salesforce/blip-image-captioning-large` | Generate text captions from images |
| **CLIP** | `openai/clip-vit-base-patch32` | Generate 512-dim embeddings for images & text |

---

## What Happens at Import

When any file does `from models import ...`, the following runs **once**:

1. **Detect device** — checks `torch.cuda.is_available()`, uses GPU if found, else CPU
2. **Load BLIP** — processor + model from `models/blip/` (local cache) or HuggingFace Hub
3. **Load CLIP** — processor + model from `models/clip/` (local cache) or HuggingFace Hub
4. All models are moved to the detected device

This takes ~10–30 seconds on first run (download) and ~5–10 seconds on subsequent runs (cached).

---

## Functions

### `load_model_safe(model_class, model_name, cache_dir, **kwargs)`

**Purpose:** Safely load a model with local-first fallback.

| Step | What It Does |
|------|-------------|
| 1 | Try loading from local cache (`local_files_only=True`) |
| 2 | If cache miss, download from HuggingFace Hub |
| 3 | Return the loaded model/processor |

**Used by:** Module init (lines 62–67) to load all 4 components (BLIP processor, BLIP model, CLIP processor, CLIP model).

---

### `generate_caption(image, max_tokens=60, num_beams=4)`

**Purpose:** Generate a natural language caption for an image using BLIP.

**Input:** PIL Image (RGB)  
**Output:** String like `"a dog sitting on a couch in a living room"`

| Step | What It Does |
|------|-------------|
| 1 | BLIP processor converts PIL image to tensor |
| 2 | BLIP model generates caption tokens using beam search (4 beams) |
| 3 | Processor decodes tokens back to text |
| 4 | Returns the caption string |

**Parameters:**
- `max_tokens=60` — Maximum words in the caption
- `num_beams=4` — Beam search width (higher = better captions, slower)

**Error handling:** Returns `"Error generating caption"` on failure.

---

### `get_image_embedding(image)`

**Purpose:** Generate a 512-dimensional vector embedding for an image using CLIP.

**Input:** PIL Image (RGB)  
**Output:** NumPy array of shape `(512,)` — a unit vector

| Step | What It Does |
|------|-------------|
| 1 | CLIP processor preprocesses the image |
| 2 | CLIP model extracts image features (no gradients) |
| 3 | Handle Transformers 5.0+ output format changes |
| 4 | Normalize to unit vector using `F.normalize()` |
| 5 | Move to CPU and convert to NumPy |

**Compatibility fixes:**
- Handles both old (tensor) and new (object) output types from `get_image_features()`
- Uses `F.normalize()` instead of `.norm()` which breaks in newer Transformers versions

**Error handling:** Returns `None` on failure.

---

### `get_text_embedding(text)` — *cached*

**Purpose:** Generate a 512-dimensional vector embedding for text using CLIP.

**Input:** String (search query)  
**Output:** NumPy array of shape `(512,)`

**Identical to `get_image_embedding`** but for text. Key difference:

- **`@lru_cache(maxsize=128)`** — caches the last 128 unique queries so repeated searches are instant (no GPU inference needed)

Same compatibility fixes as `get_image_embedding`.

---

### `get_model_info()`

**Purpose:** Return metadata about the loaded models.

**Output:**
```json
{
    "blip_model": "Salesforce/blip-image-captioning-large",
    "clip_model": "openai/clip-vit-base-patch32",
    "device": "cuda",
    "cuda_available": true
}
```

**Used by:** The `/api/stats` endpoint.

---

### `caption_to_keywords(caption: str) → List[str]`

**Purpose:** Extract meaningful keywords from a caption.

**Input:** `"a dog sitting on a couch in a living room"`  
**Output:** `["dog", "sitting", "couch", "living", "room"]`

| Step | What It Does |
|------|-------------|
| 1 | Lowercase the caption |
| 2 | Remove special characters (keep only `a-z`, `0-9`, spaces) |
| 3 | Split into words |
| 4 | Remove stopwords (a, the, is, are, with, etc. — 50+ words) |
| 5 | Remove words ≤ 2 characters |
| 6 | Deduplicate while preserving order |

**Used by:** `build_all.py` during indexing to create searchable keywords per image.

---

## File Dependencies

```
models.py
├── torch, torch.nn.functional
├── transformers (BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel)
├── functools.lru_cache
└── re (for keyword extraction)
```
