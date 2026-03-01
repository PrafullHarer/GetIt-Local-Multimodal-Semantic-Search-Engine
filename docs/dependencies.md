# Dependencies — What Each Library Does

> A breakdown of every dependency in `requirements.txt` and the built-in Python modules used across the project.

---

## requirements.txt Libraries

### Core AI / ML

| Library | Used In | What It Does |
|---------|---------|-------------|
| **torch** | `models.py` | PyTorch — the deep learning framework. Runs BLIP and CLIP models on CPU/GPU. Handles tensor operations, model inference, and gradient management. |
| **torchvision** | (indirect) | Provides image transforms and pre-trained model utilities. Required by `transformers` for vision model support. |
| **transformers** | `models.py` | HuggingFace Transformers — loads pre-trained BLIP and CLIP models and their processors. Handles tokenization, image preprocessing, and model inference. |

### Image Processing

| Library | Used In | What It Does |
|---------|---------|-------------|
| **pillow** (PIL) | `models.py`, `api.py`, `build_all.py`, `process_image.py` | Python Imaging Library — opens, resizes, converts, and saves images. Used for: reading uploaded images, converting to RGB, creating 300×300 thumbnails, and extracting image dimensions. |

### Data & Math

| Library | Used In | What It Does |
|---------|---------|-------------|
| **numpy** | `api.py` | Numerical computing — used for cosine similarity calculation between CLIP embeddings (dot product of two 512-dim vectors). Also handles array conversion from PyTorch tensors. |
| **matplotlib** | (optional) | Plotting library — available for debugging/visualization during development. Not used in the main application flow. |

### Web Server

| Library | Used In | What It Does |
|---------|---------|-------------|
| **fastapi** | `api.py` | Modern Python web framework — defines all API endpoints (`/api/search`, `/api/upload`, `/api/build`, etc.), handles request validation via Pydantic, serves the frontend, and manages background tasks. |
| **uvicorn[standard]** | `api.py` | ASGI server — runs the FastAPI app. The `[standard]` extra includes `uvloop` and `httptools` for better performance. Listens on `http://127.0.0.1:8000`. |
| **python-multipart** | `api.py` | Enables `multipart/form-data` parsing — required for the image upload endpoint (`/api/upload`) where files are sent as form data. Without this, FastAPI cannot accept file uploads. |
| **aiofiles** | `api.py` | Async file operations — used by FastAPI/Starlette to serve static files (frontend assets) without blocking the event loop. |

### Utilities

| Library | Used In | What It Does |
|---------|---------|-------------|
| **tqdm** | `build_all.py` | Progress bar — shows a live progress indicator during batch image processing: `Processing images: 45%|████▌     | 45/100 [02:15<02:45, 0.30img/s]` |

---

## Built-in Python Modules Used

These come with Python — no installation needed:

| Module | Used In | What It Does |
|--------|---------|-------------|
| **os** | all files | File path handling, directory creation, file existence checks |
| **json** | `api.py`, `build_all.py` | Read/write the index JSON files |
| **re** | `models.py` | Regex — strips special characters from captions during keyword extraction |
| **io** | `api.py` | `BytesIO` — converts uploaded file bytes into a file-like object for PIL |
| **time** | `api.py` | Measures search execution time |
| **datetime** | `api.py`, `build_all.py` | Timestamps for processing records and search history |
| **logging** | all files | Structured log output with severity levels (INFO, ERROR, WARNING) |
| **functools** | `models.py` | `lru_cache` — caches text embeddings so repeat queries skip GPU inference |
| **typing** | all files | Type hints (`List`, `Dict`, `Set`, `Optional`) for code clarity |
| **argparse** | `build_all.py` | CLI argument parsing for `--force`, `--retry`, `--status` flags |

---

## Dependency Flow

```
User Request
     │
     ▼
  uvicorn ──► fastapi ──► python-multipart (file uploads)
                │            aiofiles (static files)
                │
                ▼
           api.py uses:
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
   models.py  build_all.py  numpy
      │         │
      ▼         ▼
  transformers  pillow
      │           │
      ▼           ▼
    torch      thumbnails
   torchvision
```

---

## Install All

```bash
pip install -r requirements.txt
```

**Disk space:** ~2.5 GB (mostly PyTorch + model weights)  
**Python version:** 3.9+
