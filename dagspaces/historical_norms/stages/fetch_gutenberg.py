import json
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
            
    text = text[start_idx:end_idx].strip()

    # Normalize line endings (Windows → Unix)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove Gutenberg illustration markers
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)

    # Collapse excessive whitespace (3+ newlines → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Semantic chunking by paragraph with character-level overlap.

    Builds chunks up to *chunk_size* characters by appending whole
    paragraphs.  When a chunk is full, the next chunk starts with the
    last *overlap* characters of the previous chunk so downstream stages
    never lose context at chunk boundaries.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start next chunk with the trailing overlap from the previous
                prev = current_chunk.strip()
                current_chunk = prev[-overlap:] + "\n\n" if overlap and len(prev) > overlap else ""

            # Handle paragraphs larger than chunk_size
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 < chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            prev = current_chunk.strip()
                            current_chunk = prev[-overlap:] + " " if overlap and len(prev) > overlap else ""
                        current_chunk += sentence + " "
            else:
                current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def fetch_text_from_url(url: str) -> Optional[str]:
    """Fetch plain text from an arbitrary URL."""
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return None


def run_fetch_gutenberg(cfg: Any) -> pd.DataFrame:
    """Main execution logic for fetch_gutenberg stage.

    Supports two data config formats:
      1. gutenberg_ids: [1342, 11, 84]  -- fetches from gutenberg.org by ID
      2. text_urls:                       -- fetches from arbitrary URLs
           - id: "1984"
             url: "https://gutenberg.net.au/ebooks01/0100021.txt"

    Both can be used together in the same config.
    """
    gutenberg_ids = cfg.data.get("gutenberg_ids", [])
    text_urls = cfg.data.get("text_urls", [])
    chunk_size = cfg.data.get("chunk_size", 2000)
    overlap = cfg.data.get("overlap", 200)
    
    rows = []

    # Fetch by Gutenberg ID (standard gutenberg.org)
    for g_id in gutenberg_ids:
        print(f"[fetch_gutenberg] Fetching Gutenberg ID: {g_id}")
        raw_text = fetch_gutenberg_text(str(g_id))
        if not raw_text:
            print(f"[fetch_gutenberg] Warning: Could not fetch text for ID {g_id}")
            continue
            
        clean_text = clean_gutenberg_boilerplate(raw_text)
        chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)
        
        for i, chunk in enumerate(chunks):
            rows.append({
                "gutenberg_id": str(g_id),
                "chunk_id": i,
                "article_text": chunk,
                "chunk_size": len(chunk)
            })

    # Fetch by direct URL (for texts not on main gutenberg.org)
    for entry in text_urls:
        # Handle OmegaConf DictConfig objects (not plain dicts)
        if hasattr(entry, "get"):
            text_id = str(entry.get("id", "unknown"))
            url = str(entry.get("url", ""))
        elif isinstance(entry, str):
            # Simple string URL, use filename as ID
            url = entry
            text_id = url.rsplit("/", 1)[-1].split(".")[0]
        else:
            print(f"[fetch_gutenberg] Warning: Skipping unrecognized text_urls entry: {entry}")
            continue

        if not url:
            continue

        print(f"[fetch_gutenberg] Fetching text '{text_id}' from URL: {url}")
        raw_text = fetch_text_from_url(url)
        if not raw_text:
            print(f"[fetch_gutenberg] Warning: Could not fetch text for '{text_id}' from {url}")
            continue

        clean_text = clean_gutenberg_boilerplate(raw_text)
        chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks):
            rows.append({
                "gutenberg_id": text_id,
                "chunk_id": i,
                "article_text": chunk,
                "chunk_size": len(chunk)
            })

    df = pd.DataFrame(rows)

    # Enrich with book metadata (title, author, summary) from static JSON
    summaries_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "data", "fiction_novel_summaries.json",
    )
    if os.path.isfile(summaries_path) and len(df) > 0:
        with open(summaries_path, encoding="utf-8") as f:
            summaries = json.load(f)
        df["book_title"] = df["gutenberg_id"].map(
            lambda gid: summaries.get(str(gid), {}).get("title", "")
        )
        df["book_author"] = df["gutenberg_id"].map(
            lambda gid: summaries.get(str(gid), {}).get("author", "")
        )
        df["book_summary"] = df["gutenberg_id"].map(
            lambda gid: summaries.get(str(gid), {}).get("summary", "")
        )
        n_matched = (df["book_title"] != "").sum()
        print(f"[fetch_gutenberg] Enriched {n_matched}/{len(df)} chunks with book metadata")

    return df



