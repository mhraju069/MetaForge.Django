"""
Visual Image Search Engine using Perceptual Hashing (pHash).

Algorithm: pHash (Perceptual Hash)
- Reduces image to 8x8 DCT representation
- Converts to 64-bit binary hash (fingerprint)
- Compares using Hamming Distance: lower = more similar
- Threshold ~10: same image / very close match
- Threshold 10-25: similar style / similar product
- Threshold >25: different product

This approach is:
- Fast (milliseconds, no API calls)
- Offline / no external service needed
- Accurate for same-product matching across different photos
- Lightweight (no heavy ML models)
"""

import io
import requests
import imagehash
from PIL import Image


def compute_phash_from_url(image_url: str) -> str | None:
    """Download image and compute its pHash fingerprint string."""
    try:
        resp = requests.get(image_url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"  # Some CDNs block bots
        })
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        h = imagehash.phash(img)
        return str(h)  # e.g. "a1f3b2c4d5e6f7a8"
    except Exception as e:
        print(f"❌ [pHash] Failed to hash image {image_url}: {e}")
        return None


def compute_phash_from_bytes(image_bytes: bytes) -> str | None:
    """Compute pHash from raw image bytes (e.g., from a user upload)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        h = imagehash.phash(img)
        return str(h)
    except Exception as e:
        print(f"❌ [pHash] Failed to hash image bytes: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate Hamming Distance between two pHash strings.
    Lower = more visually similar.
    0 = identical images
    < 10 = very similar / same product
    10-25 = similar style
    > 25 = different product
    """
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2  # imagehash supports subtraction as hamming distance
    except Exception as e:
        print(f"❌ [pHash] Hamming distance error: {e}")
        return 999  # Max distance = no match


def find_best_visual_match(query_hash: str, posts_data: list, max_distance: int = 15) -> dict | None:
    """
    Find the best matching post by comparing pHash fingerprints.
    
    Args:
        query_hash: pHash of the user's image
        posts_data: list of post dicts with 'images' key (list of dicts with 'hash')
        max_distance: maximum Hamming distance to consider a match (default 15)
    
    Returns:
        Best matching post dict or None
    """
    best_match = None
    best_distance = max_distance + 1

    for post in posts_data:
        for img_data in post.get("images", []):
            stored_hash = img_data.get("hash")
            if not stored_hash:
                continue

            dist = hamming_distance(query_hash, stored_hash)
            print(f"🔍 [pHash] Post {post.get('post_id', '?')} | Distance: {dist}")

            if dist < best_distance:
                best_distance = dist
                best_match = {**post, "_visual_distance": dist}

    if best_match:
        print(f"✅ [pHash] Best visual match: Post {best_match.get('post_id')} | Distance: {best_distance}")
    else:
        print(f"⚠️ [pHash] No visual match found within threshold {max_distance}")

    return best_match
