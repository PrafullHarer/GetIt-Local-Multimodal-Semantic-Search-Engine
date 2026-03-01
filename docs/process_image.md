# process_image.py — Single Image Processor

> A lightweight utility that processes one image: generates its caption and embedding.

---

## Overview

This is a **simple wrapper** around two functions from `models.py`. It exists as a convenience utility for processing a single image outside of the full `build_all` pipeline.

**Total: 10 lines of code.**

---

## Function

### `process_image(image_path) → (caption, embedding)`

**Input:** Path to an image file (string)  
**Output:** Tuple of:
- `caption` (str) — BLIP-generated text description
- `embedding` (numpy array) — 512-dim CLIP vector

| Step | What It Does |
|------|-------------|
| 1 | Open image with PIL and convert to RGB |
| 2 | Call `generate_caption(image)` → BLIP caption |
| 3 | Call `get_image_embedding(image)` → CLIP embedding |
| 4 | Return both as a tuple |

---

## Usage

```python
from process_image import process_image

caption, embedding = process_image("data/images/photo.jpg")
print(caption)        # "a cat sleeping on a windowsill"
print(embedding.shape) # (512,)
```

---

## When to Use

| Scenario | Use This? |
|----------|----------|
| Processing a single image for testing | ✅ Yes |
| Batch processing all images | ❌ Use `build_all.py` instead |
| API upload processing | ❌ Not currently used by the API |

---

## File Dependencies

```
process_image.py
├── PIL (open image)
└── models.py (generate_caption, get_image_embedding)
```
