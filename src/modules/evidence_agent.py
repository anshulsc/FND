# src/modules/online_evidence_extractor.py
import os
import requests
import uuid
from pathlib import Path
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from difflib import SequenceMatcher

from src.config import EVIDENCE_DB_DIR

# --- Configuration ---
# Your Brave Search API key (from the original evidence_extraction.py)
BRAVE_API_KEY = "BSA1cZFR9cSOtpkalB5rDvwNEbOvZz9"

# Domains to exclude from results to reduce noise
BAD_DOMAINS = [
    "reddit.com", "quora.com", "facebook.com", "instagram.com", "x.com",
    "twitter.com", "pinterest.com", "tiktok.com", "linkedin.com",
    "medium.com", "tumblr.com", "wikipedia.org"
]

# --- Agentic Pipeline Components ---

def agent_text_search(query: str, api_key: str = BRAVE_API_KEY) -> list:
    """Agent 1: Performs a web search based on text."""
    print(f"INFO: [Agent Text Search] Searching for query: '{query[:50]}...'")
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": 15}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        print(f"INFO: [Agent Text Search] Found {len(results)} initial results.")
        return results
    except requests.exceptions.RequestException as e:
        print(f"ERROR: [Agent Text Search] Failed to query Brave API: {e}")
        return []

def agent_reverse_image_search(image_path: str) -> list:
    """Agent 2: Performs a reverse image search."""
    # This is a placeholder for a real reverse image search API (e.g., SerpApi, Google Lens API)
    # Implementing this requires an external service and API key.
    print("WARN: [Agent Reverse Image Search] This is a placeholder and did not perform a real search.")
    print("      To enable, integrate a service like SerpApi.")
    # Example structure of what a real API might return:
    # return [
    #     {"title": "Image found on this news site", "url": "http://news.com/article", "thumbnail": {"src": "url_to_image.jpg"}},
    # ]
    return []

def agent_filter_and_process(results: list, original_caption: str) -> list:
    """Agent 3: Filters, de-duplicates, and cleans the search results."""
    print("INFO: [Agent Filter] Processing and filtering search results...")
    processed = {}
    
    for r in results:
        url = r.get("url")
        if not url:
            continue

        # Rule 1: Filter out bad domains
        domain = urlparse(url).netloc
        if any(bad_domain in domain for bad_domain in BAD_DOMAINS):
            continue
            
        # Rule 2: Ensure it has a title and a thumbnail
        title = r.get("title")
        thumbnail_url = r.get("thumbnail", {}).get("src")
        if not title or not thumbnail_url:
            continue
            
        # Rule 3: Avoid results that are too similar to the original query
        if SequenceMatcher(None, original_caption.lower(), title.lower()).ratio() > 0.9:
            continue

        # Rule 4: De-duplicate based on URL
        if url not in processed:
            processed[url] = {"title": title.strip(), "image_url": thumbnail_url}
            
    filtered_list = list(processed.values())
    print(f"INFO: [Agent Filter] {len(filtered_list)} results remain after filtering.")
    return filtered_list

def agent_save_evidence(item: dict) -> Path:
    """Agent 4: Downloads an image and saves the evidence pair to the database."""
    try:
        # Create a unique directory for this piece of evidence
        evidence_id = f"online_{uuid.uuid4().hex[:8]}"
        save_dir = EVIDENCE_DB_DIR / evidence_id
        save_dir.mkdir(exist_ok=True)

        # Download the image
        response = requests.get(item['image_url'], timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Save image (e.g., as evidence.jpg)
        image_path = save_dir / "evidence.jpg"
        img.save(image_path)
        
        # Save caption
        caption_path = save_dir / "caption.txt"
        caption_path.write_text(item['title'], encoding='utf-8')
        
        return save_dir
    except Exception as e:
        print(f"ERROR: [Agent Saver] Failed to download/save evidence for '{item['title']}'. Reason: {e}")
        return None

# --- Main Pipeline Orchestrator ---

def run_online_extraction_pipeline(query_image_path: str, query_caption: str) -> dict:
    """
    Orchestrates the agentic pipeline to find and archive new evidence.
    """
    print("\n--- Starting Online Evidence Extraction Pipeline ---")
    
    # Step 1: Run search agents
    text_results = agent_text_search(query_caption)
    image_results = agent_reverse_image_search(query_image_path)
    
    # Step 2: Combine and filter results
    all_results = text_results + image_results
    if not all_results:
        print("--- Pipeline Finished: No initial search results found. ---")
        return {"found_new": 0, "saved_paths": []}
        
    clean_candidates = agent_filter_and_process(all_results, query_caption)
    
    # Step 3: Save the filtered evidence
    saved_paths = []
    if not clean_candidates:
        print("--- Pipeline Finished: No suitable candidates after filtering. ---")
        return {"found_new": 0, "saved_paths": []}

    print(f"INFO: Attempting to save {len(clean_candidates)} evidence candidates...")
    for item in clean_candidates:
        saved_path = agent_save_evidence(item)
        if saved_path:
            saved_paths.append(str(saved_path))
            
    summary = {
        "found_new": len(saved_paths),
        "saved_paths": saved_paths
    }
    
    print(f"--- Pipeline Finished: Successfully saved {summary['found_new']} new evidence items. ---")
    return summary

if __name__ == "__main__":
    # This allows for standalone testing of the pipeline
    # Create dummy files for the test
    dummy_dir = Path("./dummy_query")
    dummy_dir.mkdir(exist_ok=True)
    dummy_image_path = dummy_dir / "test_img.jpg"
    dummy_caption_path = dummy_dir / "test_cap.txt"
    Image.new('RGB', (100, 100)).save(dummy_image_path) # Create a blank image
    dummy_caption_path.write_text("Supreme Court extends interim bail for Kerala actor")
    
    print("Running standalone test...")
    run_online_extraction_pipeline(str(dummy_image_path), dummy_caption_path.read_text())