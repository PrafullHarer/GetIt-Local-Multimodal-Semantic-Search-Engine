"""
python build_all.py           # Normal incremental run
python build_all.py --force   # Reprocess everything
python build_all.py --retry   # Retry only failed images
python build_all.py --status  # Show index status
"""
import os
import json
import re
import logging
from typing import List, Dict, Set
from datetime import datetime

from PIL import Image
from models import generate_caption, get_image_embedding, caption_to_keywords
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
THUMBNAIL_DIR = os.path.join(PROJECT_ROOT, "data", "thumbnails")
INDEX_JSON = os.path.join(PROJECT_ROOT, "data", "index.json")
INDEX_WITH_KW_JSON = os.path.join(PROJECT_ROOT, "data", "index_with_keywords.json")

# Create directories
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# Thumbnail size
THUMBNAIL_SIZE = (300, 300)


def load_existing_index(path: str) -> Dict[str, dict]:
    """
    Load existing index and return as dict keyed by filename.
    
    Returns:
        Dict mapping filename -> entry data
    """
    if not os.path.exists(path):
        return {}
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert list to dict for O(1) lookup
        return {item["filename"]: item for item in data}
    except Exception as e:
        logger.warning(f"Could not load existing index: {e}")
        return {}


def is_processed(entry: dict) -> bool:
    """
    Check if an entry has been fully processed.
    
    An entry is considered processed if:
    - It has the 'processed' flag set to True
    - It has a valid embedding (non-empty list)
    - It has a caption
    """
    if not entry:
        return False
    
    has_flag = entry.get("processed", False) is True
    has_embedding = isinstance(entry.get("embedding"), list) and len(entry.get("embedding", [])) > 0
    has_caption = bool(entry.get("caption"))
    
    return has_flag and has_embedding and has_caption


def create_thumbnail(image_path: str, filename: str) -> bool:
    """
    Create a thumbnail for faster web loading.
    
    Returns:
        bool: Success status
    """
    try:
        thumb_path = os.path.join(THUMBNAIL_DIR, filename)
        
        # Skip if thumbnail already exists and is valid
        if os.path.exists(thumb_path):
            # Verify thumbnail is not corrupted
            try:
                img = Image.open(thumb_path)
                img.verify()
                return True
            except:
                logger.info(f"Regenerating corrupted thumbnail: {filename}")
        
        img = Image.open(image_path)
        img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (for PNG with transparency)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        img.save(thumb_path, "JPEG", quality=85, optimize=True)
        return True
    except Exception as e:
        logger.error(f"Error creating thumbnail for {filename}: {e}")
        return False


def get_image_metadata(image_path: str) -> dict:
    """
    Extract basic image metadata.
    """
    try:
        img = Image.open(image_path)
        stat = os.stat(image_path)
        
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metadata for {image_path}: {e}")
        return {}


def cleanup_deleted_images(existing_index: Dict[str, dict], current_files: Set[str]) -> Dict[str, dict]:
    """
    Remove entries for images that no longer exist in the folder.
    
    Returns:
        Cleaned index dict
    """
    deleted = set(existing_index.keys()) - current_files
    
    if deleted:
        logger.info(f"Removing {len(deleted)} deleted images from index")
        for filename in deleted:
            del existing_index[filename]
            # Also remove thumbnail if exists
            thumb_path = os.path.join(THUMBNAIL_DIR, filename)
            if os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except:
                    pass
    
    return existing_index


def build_all(force_reprocess: bool = False):
    """
    Enhanced pipeline with incremental processing:
    - Only processes new/unprocessed images
    - Adds 'processed' flag to track status
    - Cleans up deleted images
    - Thumbnail generation
    - Metadata extraction
    
    Args:
        force_reprocess: If True, reprocess all images regardless of flag
    """
    
    if not os.path.isdir(IMAGE_DIR):
        logger.error(f"Image folder not found: {IMAGE_DIR}")
        return

    # Get list of image files
    file_list = sorted(os.listdir(IMAGE_DIR))
    image_files = [
        f for f in file_list 
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))
    ]
    current_files = set(image_files)
    
    logger.info(f"Found {len(image_files)} images in folder")

    # Load existing index
    existing_index = load_existing_index(INDEX_WITH_KW_JSON)
    if not existing_index:
        existing_index = load_existing_index(INDEX_JSON)
    
    logger.info(f"Loaded {len(existing_index)} existing entries from index")

    # Cleanup deleted images
    existing_index = cleanup_deleted_images(existing_index, current_files)

    # Determine which images need processing
    if force_reprocess:
        to_process = image_files
        logger.info("Force reprocess enabled - processing all images")
    else:
        to_process = [
            f for f in image_files 
            if not is_processed(existing_index.get(f))
        ]
    
    already_processed = len(image_files) - len(to_process)
    
    if already_processed > 0:
        logger.info(f"Skipping {already_processed} already processed images")
    
    if not to_process:
        logger.info("No new images to process!")
        # Still save to ensure cleanup is persisted
        save_index(existing_index)
        return

    logger.info(f"Processing {len(to_process)} new/unprocessed images")

    # Process new images
    failed_images = []
    
    for filename in tqdm(to_process, desc="Processing images", unit="img"):
        try:
            image_path = os.path.join(IMAGE_DIR, filename)
            
            # Open and validate image
            image = Image.open(image_path).convert("RGB")
            
            # Create thumbnail
            create_thumbnail(image_path, filename)
            
            # Generate caption
            caption = generate_caption(image)
            
            # Generate embedding
            embedding = get_image_embedding(image)
            
            if embedding is None:
                raise ValueError("Failed to generate embedding")
            
            # Extract keywords
            keywords = caption_to_keywords(caption)
            
            # Get metadata
            metadata = get_image_metadata(image_path)
            
            # Create/update entry with processed flag
            existing_index[filename] = {
                "filename": filename,
                "caption": caption,
                "embedding": embedding.tolist(),
                "keywords": keywords,
                "metadata": metadata,
                "processed": True,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            failed_images.append(filename)
            
            # Mark as not processed if it fails
            if filename in existing_index:
                existing_index[filename]["processed"] = False
                existing_index[filename]["error"] = str(e)
            continue

    # Save index
    save_index(existing_index)

    # Summary
    print("\n" + "="*50)
    print(f"📊 Processing Summary")
    print("="*50)
    print(f"  Total images in folder:  {len(image_files)}")
    print(f"  Already processed:       {already_processed}")
    print(f"  Newly processed:         {len(to_process) - len(failed_images)}")
    if failed_images:
        print(f"  Failed:                  {len(failed_images)}")
        print(f"    → {', '.join(failed_images[:5])}")
        if len(failed_images) > 5:
            print(f"    → ... and {len(failed_images) - 5} more")
    print(f"  Thumbnails dir:          {THUMBNAIL_DIR}")
    print("="*50 + "\n")


def save_index(index_dict: Dict[str, dict]):
    """
    Save index to both JSON files.
    """
    # Convert dict back to list
    entries = list(index_dict.values())
    
    # Ensure data directory exists
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    
    # Save basic index (without keywords for backward compatibility)
    basic_entries = []
    for item in entries:
        basic_entry = {
            "filename": item["filename"],
            "caption": item.get("caption", ""),
            "embedding": item.get("embedding", []),
            "metadata": item.get("metadata", {}),
            "processed": item.get("processed", False),
            "processed_at": item.get("processed_at", "")
        }
        basic_entries.append(basic_entry)
    
    with open(INDEX_JSON, "w") as f:
        json.dump(basic_entries, f, indent=2)
    logger.info(f"✓ Saved basic index to: {INDEX_JSON}")

    # Save extended index with keywords
    with open(INDEX_WITH_KW_JSON, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"✓ Saved extended index to: {INDEX_WITH_KW_JSON}")


def reprocess_failed():
    """
    Reprocess only images that previously failed.
    """
    existing_index = load_existing_index(INDEX_WITH_KW_JSON)
    
    failed = [
        filename for filename, entry in existing_index.items()
        if entry.get("processed") is False or entry.get("error")
    ]
    
    if not failed:
        logger.info("No failed images to reprocess")
        return
    
    logger.info(f"Found {len(failed)} failed images to reprocess")
    
    # Clear their processed status to force reprocessing
    for filename in failed:
        if filename in existing_index:
            existing_index[filename]["processed"] = False
            if "error" in existing_index[filename]:
                del existing_index[filename]["error"]
    
    # Save and run build
    save_index(existing_index)
    build_all()


def get_index_status():
    """
    Print current index status without processing.
    """
    existing_index = load_existing_index(INDEX_WITH_KW_JSON)
    
    if not existing_index:
        print("No index found.")
        return
    
    processed = sum(1 for e in existing_index.values() if is_processed(e))
    failed = sum(1 for e in existing_index.values() if e.get("error"))
    pending = len(existing_index) - processed
    
    # Check for new files not in index
    if os.path.isdir(IMAGE_DIR):
        file_list = sorted(os.listdir(IMAGE_DIR))
        image_files = set(
            f for f in file_list 
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))
        )
        new_files = image_files - set(existing_index.keys())
    else:
        new_files = set()
    
    print("\n" + "="*50)
    print("📊 Index Status")
    print("="*50)
    print(f"  Total in index:    {len(existing_index)}")
    print(f"  Processed:         {processed}")
    print(f"  Pending:           {pending}")
    print(f"  Failed:            {failed}")
    print(f"  New (not indexed): {len(new_files)}")
    if new_files and len(new_files) <= 5:
        print(f"    → {', '.join(new_files)}")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build image search index")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reprocess all images"
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry only failed images"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show index status without processing"
    )
    
    args = parser.parse_args()
    
    if args.status:
        get_index_status()
    elif args.retry:
        reprocess_failed()
    else:
        build_all(force_reprocess=args.force)