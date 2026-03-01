import os
import json
import time
import logging
from typing import List, Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

# Import build_all function
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Import build_all function
from build_all import build_all

from models import (
    generate_caption, 
    get_image_embedding, 
    get_text_embedding,
    get_model_info,
    caption_to_keywords
)

# Initialize FastAPI
app = FastAPI(
    title="Image Search AI",
    description="Semantic image search using BLIP + CLIP",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
THUMBNAIL_DIR = os.path.join(PROJECT_ROOT, "data", "thumbnails")
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "index_with_keywords.json")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Create directories
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# Load index on startup
index_data = []
search_history = []

def load_index():
    """Load the image index."""
    global index_data
    try:
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, 'r') as f:
                index_data = json.load(f)
            logger.info(f"Loaded {len(index_data)} images from index")
        else:
            logger.warning("No index found. Run build_all.py first.")
            index_data = []
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        index_data = []

@app.on_event("startup")
async def startup_event():
    """Load index on startup."""
    load_index()


# Pydantic models
class SearchResult(BaseModel):
    filename: str
    caption: str
    keywords: List[str]
    clip_score: float
    keyword_score: float
    final_score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float


class StatsResponse(BaseModel):
    total_images: int
    total_keywords: int
    unique_keywords: int
    model_info: dict


# Helper functions
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized cosine similarity."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def compute_search_scores(
    query_embed: np.ndarray,
    query_keywords: set,
    index_entries: list,
    clip_weight: float = 0.8,
    keyword_weight: float = 0.2
) -> list:
    """Compute search scores with configurable weights."""
    results = []
    
    for item in index_entries:
        # CLIP similarity
        emb = np.array(item["embedding"], dtype=np.float32)
        clip_score = cosine_similarity(query_embed, emb)
        
        # Keyword overlap
        item_keywords = set(item.get("keywords", []))
        if query_keywords and item_keywords:
            overlap = len(query_keywords & item_keywords)
            keyword_score = overlap / len(query_keywords) if query_keywords else 0.0
        else:
            keyword_score = 0.0
        
        # Combined score
        final_score = clip_weight * clip_score + keyword_weight * keyword_score
        
        results.append({
            "filename": item["filename"],
            "caption": item.get("caption", ""),
            "keywords": item.get("keywords", []),
            "clip_score": float(clip_score),
            "keyword_score": float(keyword_score),
            "final_score": float(final_score),
            "metadata": item.get("metadata", {})
        })
    
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Image Search AI</h1><p>Frontend not found. Create frontend/index.html</p>")




@app.get("/api/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum score threshold"),
    clip_weight: float = Query(0.8, ge=0.0, le=1.0, description="CLIP weight"),
    keyword_weight: float = Query(0.2, ge=0.0, le=1.0, description="Keyword weight")
):
    start_time = time.time()
    
    if not index_data:
        raise HTTPException(status_code=503, detail="Index not loaded. Run build_all.py first.")
    
    try:
        query_embed = get_text_embedding(query)
        if query_embed is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        query_keywords = set(query.lower().split())
        
        results = compute_search_scores(
            query_embed,
            query_keywords,
            index_data,
            clip_weight,
            keyword_weight
        )
        
        results = [r for r in results if r["final_score"] >= min_score]
        results = results[:top_k]
        
        search_time = time.time() - start_time
        
        search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results)
        })
        if len(search_history) > 100:
            search_history.pop(0)
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time=search_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Upload multiple images. 
    NOTE: This only saves them to disk. It does NOT process them immediately 
    to prevent timeouts on large batches. 
    Use the 'Build Index' button after uploading.
    """
    saved_files = []
    try:
        for file in files:
            # Validate file type
            if not file.content_type.startswith("image/"):
                continue
            
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Save image
            filename = file.filename
            image_path = os.path.join(IMAGE_DIR, filename)
            image.save(image_path)
            
            saved_files.append(filename)
            
        return {
            "success": True,
            "count": len(saved_files),
            "files": saved_files,
            "message": f"Successfully saved {len(saved_files)} images. Please click 'Build Index' to process them."
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/build")
async def trigger_build(background_tasks: BackgroundTasks):
    """
    Trigger the build_all process in the background.
    """
    def run_build_and_reload():
        logger.info("Starting background build process...")
        build_all(force_reprocess=False)
        load_index() # Reload memory after build finishes
        logger.info("Background build finished and index reloaded.")

    background_tasks.add_task(run_build_and_reload)
    
    return {
        "success": True, 
        "message": "Index build started in background. Check 'Stats' or server logs for progress."
    }


@app.get("/api/image/{filename}")
async def get_image(filename: str, thumbnail: bool = False):
    try:
        if thumbnail:
            path = os.path.join(THUMBNAIL_DIR, filename)
        else:
            path = os.path.join(IMAGE_DIR, filename)
        
        if not os.path.exists(path):
            # Fallback to full image if thumbnail missing
            if thumbnail and os.path.exists(os.path.join(IMAGE_DIR, filename)):
                path = os.path.join(IMAGE_DIR, filename)
            else:
                raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(path)
        
    except Exception as e:
        logger.error(f"Image retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    try:
        all_keywords = []
        for item in index_data:
            all_keywords.extend(item.get("keywords", []))
        
        return StatsResponse(
            total_images=len(index_data),
            total_keywords=len(all_keywords),
            unique_keywords=len(set(all_keywords)),
            model_info=get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_search_history():
    return {"history": search_history[-20:]}


@app.post("/api/reload")
async def reload_index_endpoint():
    load_index()
    return {
        "success": True,
        "message": f"Reloaded {len(index_data)} images"
    }


if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)