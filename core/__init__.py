"""
Core package for the emotional recovery assistant.

Heavy optional dependencies for vector search, facial expression recognition,
speech interaction, and transformer models are imported only when needed.
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
