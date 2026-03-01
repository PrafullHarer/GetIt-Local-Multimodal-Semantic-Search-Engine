# build_all.py — Image Indexing Pipeline

> Scans images, generates captions and embeddings, creates thumbnails, and saves the searchable index.

---

## Overview

This is the **indexing engine**. It processes every image in `data/images/` through the AI pipeline and builds the JSON index that powers search. Supports incremental processing (only new images), retry of failures, and status reporting.

**Run modes:**
```bash
python build_all.py           # Normal incremental run
python build_all.py --force   # Reprocess everything
python build_all.py --retry   # Retry only failed images
python build_all.py --status  # Show index status
```

---

## The Build Pipeline

For each **unprocessed** image, `build_all()` runs these steps:

```
Image File
  │
  ├─ 1. Open with PIL → convert to RGB
  │
  ├─ 2. Create Thumbnail (300×300, JPEG, 85% quality)
  │     └─ Saved to data/thumbnails/{filename}
  │
  ├─ 3. Generate Caption (BLIP)
  │     └─ "a dog sitting on a couch in a living room"
  │
  ├─ 4. Generate Embedding (CLIP)
  │     └─ 512-dimensional float32 vector
  │
  ├─ 5. Extract Keywords
  │     └─ ["dog", "sitting", "couch", "living", "room"]
  │
  ├─ 6. Extract Metadata
  │     └─ {width, height, format, mode, size_bytes, modified}
  │
  └─ 7. Save Entry to Index
        └─ {filename, caption, embedding, keywords, metadata, processed: true}
```

---

## Functions

### `build_all(force_reprocess=False)` — Main Entry Point

The primary indexing function. Orchestrates the entire pipeline.

| Step | What It Does |
|------|-------------|
| 1 | Scan `data/images/` for supported files (jpg, jpeg, png, webp, bmp, gif) |
| 2 | Load existing index from `index_with_keywords.json` |
| 3 | Clean up entries for deleted images (`cleanup_deleted_images`) |
| 4 | Filter out already processed images (skip them) |
| 5 | For each unprocessed image: caption → embed → keywords → thumbnail → metadata |
| 6 | Save updated index to both JSON files |
| 7 | Print processing summary |

**Incremental behavior:**
- Default: only processes images without `processed: true` flag
- `force_reprocess=True`: reprocesses every image regardless of flag

---

### `load_existing_index(path) → Dict[str, dict]`

Loads the JSON index and converts it from a list to a dict keyed by filename for O(1) lookups.

**Input:** Path to JSON file  
**Output:** `{"photo1.jpg": {entry...}, "photo2.jpg": {entry...}}`

---

### `is_processed(entry) → bool`

Checks if an index entry is fully processed. An entry is considered processed if **all three** conditions are true:

| Check | What It Verifies |
|-------|-----------------|
| `processed` flag | Must be `True` |
| `embedding` | Must be a non-empty list |
| `caption` | Must be a non-empty string |

---

### `create_thumbnail(image_path, filename) → bool`

Creates a 300×300 JPEG thumbnail for fast web loading.

| Step | What It Does |
|------|-------------|
| 1 | Check if thumbnail already exists and is valid |
| 2 | If exists, verify it's not corrupted (`img.verify()`) |
| 3 | If missing or corrupted, create new thumbnail |
| 4 | Convert RGBA/P images to RGB (for PNG transparency) |
| 5 | Save as JPEG with 85% quality and optimization |

---

### `get_image_metadata(image_path) → dict`

Extracts basic metadata from an image file:

```json
{
    "width": 1920,
    "height": 1080,
    "format": "JPEG",
    "mode": "RGB",
    "size_bytes": 245760,
    "modified": "2025-03-01T12:00:00"
}
```

---

### `cleanup_deleted_images(existing_index, current_files) → Dict`

Removes index entries for images that no longer exist on disk. Also deletes orphaned thumbnails.

---

### `save_index(index_dict)`

Saves the index to two JSON files:

| File | Contents | Purpose |
|------|----------|---------|
| `data/index.json` | Entries without keywords | Backward compatibility |
| `data/index_with_keywords.json` | Full entries with keywords | Primary index |

---

### `reprocess_failed()`

Finds all entries with `processed: false` or an `error` field, clears their error status, and runs `build_all()` again — only these images will be picked up for reprocessing.

---

### `get_index_status()`

Prints a summary of the current index without doing any processing:

```
==================================================
📊 Index Status
==================================================
  Total in index:    150
  Processed:         147
  Pending:           2
  Failed:            1
  New (not indexed): 5
==================================================
```

---

## Index Entry Format

Each image produces one entry in the JSON index:

```json
{
    "filename": "photo.jpg",
    "caption": "a dog sitting on a couch in a living room",
    "embedding": [0.023, -0.015, 0.042, ...],  // 512 floats
    "keywords": ["dog", "sitting", "couch", "living", "room"],
    "metadata": {
        "width": 1920,
        "height": 1080,
        "format": "JPEG",
        "mode": "RGB",
        "size_bytes": 245760,
        "modified": "2025-03-01T12:00:00"
    },
    "processed": true,
    "processed_at": "2025-03-01T12:00:05"
}
```

---

## CLI Usage

```bash
# First-time setup or adding new images
python build_all.py

# Force reprocess everything (e.g., after model update)
python build_all.py --force

# Retry images that failed last time
python build_all.py --retry

# Check status without processing
python build_all.py --status
```

---

## File Dependencies

```
build_all.py
├── PIL (image processing, thumbnails)
├── tqdm (progress bar)
├── models.py (generate_caption, get_image_embedding, caption_to_keywords)
└── json, os, datetime, logging
```
