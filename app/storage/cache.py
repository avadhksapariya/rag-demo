import json
import time
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

from app.core.config import CACHE_PATH
from app.src.ingestion.embeddings import get_embeddings

CACHE_FILE = CACHE_PATH / "semantic_cache.json"

DEFAULT_TTL_SECONDS = 24 * 3600  # 24 hours expiration
MAX_CACHE_ENTRIES = 50  # Keep at most 50 recent entries


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class SemanticCache:

    def __init__(
        self,
        threshold: float = 0.92,
        cache_path: Path = CACHE_FILE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = MAX_CACHE_ENTRIES,
    ):
        self.threshold = threshold
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._embeddings = None
        self.cache_entries = self._load_and_prune_cache()

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    # Loads entries from disk and evicts expired records.
    def _load_and_prune_cache(self) -> list[dict]:
        if not self.cache_path.exists():
            return []

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw_entries = json.load(f)
        except Exception as e:
            print(f"[CACHE] Warning: Failed to read cache: {e}")
            return []

        current_time = time.time()

        # Keep only entries that have not exceeded the TTL
        valid_entries = [
            e
            for e in raw_entries
            if current_time - e.get("timestamp", 0) < self.ttl_seconds
        ]

        # Save back if stale records were pruned
        if len(valid_entries) < len(raw_entries):
            self._save_to_disk(valid_entries)

        return valid_entries

    # Persists cache to disk without bloated indentation.
    def _save_to_disk(self, entries: list[dict]):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    entries,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda o: (
                        float(o) if isinstance(o, (np.floating, np.integer)) else str(o)
                    ),
                )
        except Exception as e:
            print(f"[CACHE] Warning: Failed to save cache: {e}")

    # Checks if a semantically equivalent query already exists in the cache.
    def lookup(
        self,
        query: str,
        selected_file: Optional[str] = None,
    ) -> Optional[Tuple[dict, float]]:
        if not self.cache_entries:
            return None

        current_time = time.time()

        # Filter out expired entries on the fly
        self.cache_entries = [
            e
            for e in self.cache_entries
            if current_time - e.get("timestamp", 0) < self.ttl_seconds
        ]

        query_vec = self.embeddings.embed_query(query)
        best_match = None
        best_score = -1.0

        for entry in self.cache_entries:
            if entry.get("selected_file") != selected_file:
                continue

            score = _cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.threshold and best_match:
            # Update last accessed time for LRU eviction order
            best_match["timestamp"] = current_time
            return best_match["data"], float(best_score)
        return None

    # Stores a new query and response payload in the cache.
    def store(self, query: str, data: dict, selected_file: Optional[str] = None):
        query_vec = self.embeddings.embed_query(query)
        clean_vec = [
            round(float(x), 4) for x in query_vec
        ]  # round float precision to reduce JSON size

        new_entry = {
            "query": query,
            "selected_file": selected_file,
            "timestamp": time.time(),
            "embedding": clean_vec,
            "data": data,
        }

        self.cache_entries.append(new_entry)

        # Enforce max limit: retain only newest max_entries
        if len(self.cache_entries) > self.max_entries:
            self.cache_entries = self.cache_entries[-self.max_entries]

        self._save_to_disk(self.cache_entries)

    # Manually flushes the entire cache.
    def clear(self):
        self.cache_entries = []

        if self.cache_path.exists():
            self.cache_path.unlink()

        # Remove the cache directory if it is empty
        if self.cache_path.parent.exists():
            try:
                self.cache_path.parent.rmdir()
            except OSError:
                # Directory is not empty, so leave it intact
                pass

        print("🧹 [Semantic Cache]: Flushed completely.")


# Global singleton instance
semantic_cache = SemanticCache(threshold=0.90)
