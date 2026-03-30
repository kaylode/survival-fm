import timeit
import hashlib
from typing import Any, Dict, Optional


class Timer:
    """Context manager for timing code execution."""

    _trackers: Dict[str, float] = {}

    def _create_hash(self, name: Optional[str]) -> str:
        """Create a unique hash from name and parameters."""
        if not name:
            return "default"
        # Create short hash (first 8 chars of MD5)
        hash_obj = hashlib.md5(name.encode())
        return f"{name or 'timer'}_{hash_obj.hexdigest()[:8]}"

    def __enter__(self, **kwargs):
        self.start('default')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop('default')
        return False  # Don't suppress exceptions

    def start(self, name: Optional[str] = None):
        hash_key = self._create_hash(name)
        if hash_key not in self._trackers:
            self._trackers[hash_key] = 0
        self.start_time = timeit.default_timer()

    def stop(self, name: Optional[str] = None) -> float:
        self.elapsed = timeit.default_timer() - self.start_time
        hash_key = self._create_hash(name)
        self._trackers[hash_key] += self.elapsed
        return self.elapsed