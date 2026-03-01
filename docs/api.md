# api.py — FastAPI Server & Endpoints

> The web server. Handles HTTP requests, serves the frontend, orchestrates search, upload, and indexing.

---

## Overview

This is the **entry point** of the application. It creates a FastAPI server that:
- Serves the frontend UI at `/`
- Exposes REST API endpoints under `/api/`
- Loads the image index into memory at startup
- Coordinates between the frontend and the AI models

**Run with:** `python app/api.py` → starts on `http://127.0.0.1:8000`

---

## Startup Sequence

```
python app/api.py
  │
  ├─ 1. Import models.py → loads BLIP + CLIP into GPU/CPU memory
  ├─ 2. Import build_all.py → makes build_all() available
  ├─ 3. Create FastAPI app with CORS middleware
  ├─ 4. @startup_event → load_index()
  │      └─ Read data/index_with_keywords.json → index_data[] (in memory)
  ├─ 5. Mount frontend/ as static files
  └─ 6. Uvicorn starts listening on port 8000
```

---

## Data Models (Pydantic)

### `SearchResult`
| Field | Type | Description |
|-------|------|-------------|
| `filename` | str | Image filename |
| `caption` | str | BLIP-generated caption |
| `keywords` | List[str] | Extracted keywords |
| `clip_score` | float | CLIP cosine similarity (0–1) |
| `keyword_score` | float | Keyword overlap ratio (0–1) |
| `final_score` | float | Weighted blend of both scores |
| `metadata` | dict (optional) | Image dimensions, format, size |

### `SearchResponse`
| Field | Type | Description |
|-------|------|-------------|
| `query` | str | The original search query |
| `results` | List[SearchResult] | Ranked results |
| `total_results` | int | Number of results returned |
| `search_time` | float | Time taken in seconds |

### `StatsResponse`
| Field | Type | Description |
|-------|------|-------------|
| `total_images` | int | Images in the index |
| `total_keywords` | int | Total keyword count |
| `unique_keywords` | int | Distinct keywords |
| `model_info` | dict | Model names, device, CUDA status |

---

## API Endpoints

### `GET /` — Serve Frontend

Returns `frontend/index.html` as the main page. Falls back to a simple HTML message if the file is missing.

---

### `GET /api/search` — Search Images

The core search endpoint. Finds images matching a natural language query.

**Query Parameters:**
| Param | Default | Range | Description |
|-------|---------|-------|-------------|
| `query` | required | min 1 char | The search text |
| `top_k` | 10 | 1–100 | Max results to return |
| `min_score` | 0.0 | 0.0–1.0 | Minimum score threshold |
| `clip_weight` | 0.8 | 0.0–1.0 | Weight for CLIP similarity |
| `keyword_weight` | 0.2 | 0.0–1.0 | Weight for keyword overlap |

**What happens internally:**

| Step | What It Does |
|------|-------------|
| 1 | Generate text embedding for query via CLIP (`get_text_embedding`) |
| 2 | Split query into keyword set |
| 3 | Score every image in the index (`compute_search_scores`) |
| 4 | Filter by `min_score` |
| 5 | Sort by `final_score` descending |
| 6 | Return top `top_k` results |
| 7 | Log to `search_history` (max 100 entries) |

---

### `POST /api/upload` — Upload Images

Accepts multiple image files and saves them to `data/images/`.

**Important:** Upload does **NOT** process images (no AI inference). The user must click "Build Index" separately to trigger captioning and embedding.

| Step | What It Does |
|------|-------------|
| 1 | Validate each file is an image (`content_type` check) |
| 2 | Open with PIL, convert to RGB |
| 3 | Save to `data/images/{filename}` |
| 4 | Return count of saved files |

---

### `POST /api/build` — Trigger Index Build

Starts the `build_all()` function in a **background task** (non-blocking).

| Step | What It Does |
|------|-------------|
| 1 | Enqueue `build_all()` as a FastAPI background task |
| 2 | After build completes, automatically call `load_index()` to refresh memory |
| 3 | Return immediately with success message |

---

### `GET /api/image/{filename}` — Serve Images

Returns an image file from disk.

| Param | Effect |
|-------|--------|
| `thumbnail=false` | Serve from `data/images/` (original) |
| `thumbnail=true` | Serve from `data/thumbnails/` (300×300 JPEG) |

Falls back to original if thumbnail is missing.

---

### `GET /api/stats` — Index Statistics

Returns counts of images, keywords, and model info. Reads from the in-memory `index_data`.

---

### `GET /api/history` — Search History

Returns the last 20 search queries with timestamps and result counts.

---

### `POST /api/reload` — Reload Index

Re-reads `index_with_keywords.json` from disk into memory. Useful after manual edits or external builds.

---

## Helper Functions

### `cosine_similarity(a, b)`

Computes dot product of two unit vectors. Since embeddings are pre-normalized by `F.normalize()` in models.py, `dot product = cosine similarity`.

### `compute_search_scores(query_embed, query_keywords, index_entries, clip_weight, keyword_weight)`

Scores every image in the index:

```
final_score = clip_weight × cosine_sim(query_embed, image_embed)
            + keyword_weight × (|query_kw ∩ image_kw| / |query_kw|)
```

Returns results sorted by `final_score` descending.

---

## File Dependencies

```
api.py
├── fastapi, uvicorn
├── numpy (cosine similarity)
├── pydantic (data models)
├── PIL (image upload handling)
├── models.py (AI inference)
└── build_all.py (index building)
```
