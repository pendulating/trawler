import requests
import re
import pandas as pd
from typing import List, Dict, Any, Optional
import os

def fetch_gutenberg_text(gutenberg_id: str) -> Optional[str]:
    """Fetch text from Project Gutenberg mirror."""
    # Common URL patterns for Project Gutenberg
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue
    return None

def clean_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg headers and footers."""
    # Common start/end markers
    start_markers = [
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*",
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*",
    ]
    end_markers = [
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*",
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*",
    ]
    
    start_idx = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            break
            
    end_idx = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_idx = match.start()
            break
            
    return text[start_idx:end_idx].strip()

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Simple semantic chunking by paragraph/sentence."""
    # Split by double newline (paragraphs) first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Handle paragraphs larger than chunk_size by splitting at sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def run_fetch_gutenberg(cfg: Any) -> pd.DataFrame:
    """Main execution logic for fetch_gutenberg stage."""
    gutenberg_ids = cfg.data.get("gutenberg_ids", [])
    chunk_size = cfg.data.get("chunk_size", 2000)
    overlap = cfg.data.get("overlap", 200)
    
    rows = []
    for g_id in gutenberg_ids:
        print(f"Fetching Gutenberg ID: {g_id}")
        raw_text = fetch_gutenberg_text(str(g_id))
        if not raw_text:
            print(f"Warning: Could not fetch text for ID {g_id}")
            continue
            
        clean_text = clean_gutenberg_boilerplate(raw_text)
        chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)
        
        for i, chunk in enumerate(chunks):
            rows.append({
                "gutenberg_id": str(g_id),
                "chunk_id": i,
                "article_text": chunk, # Keep naming consistent with uair orchestrator if possible
                "chunk_size": len(chunk)
            })
            
    return pd.DataFrame(rows)



