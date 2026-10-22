"""
Central configuration for the multi-agent document comparison system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"

# Keep model downloads in the project rather than the user's protected home.
# Configure these before importing Hugging Face or torch: they cache defaults
# when their modules are imported.
CACHE_DIR = PROJECT_ROOT.parent / ".cache"
MODEL_CACHE_DIR = CACHE_DIR / "huggingface" / "hub"
for cache_name, cache_path in {
    "XDG_CACHE_HOME": CACHE_DIR,
    "HF_HOME": CACHE_DIR / "huggingface",
    "HF_HUB_CACHE": MODEL_CACHE_DIR,
    "HF_XET_CACHE": CACHE_DIR / "huggingface" / "xet",
    "TORCH_HOME": CACHE_DIR / "torch",
    "SENTENCE_TRANSFORMERS_HOME": MODEL_CACHE_DIR,
}.items():
    os.environ.setdefault(cache_name, str(cache_path))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding configuration
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # MiniLM output dimension

# Chunking parameters
TEXT_CHUNK_SIZE = 512  # tokens
TEXT_CHUNK_OVERLAP = 50  # tokens

# Similarity parameters
TOP_K_MATCHES = 10  # Number of similar chunks to retrieve

# Modality weights (Phase 2: all modalities)
# These weights must sum to 1.0
MODALITY_WEIGHTS = {
    "text": 0.35,
    "table": 0.25,
    "image": 0.20,     # Phase 2
    "layout": 0.10,    # Phase 2
    "metadata": 0.10   # Phase 2
}

# Phase 1 only weights (backward compatibility)
MODALITY_WEIGHTS_PHASE1 = {
    "text": 0.60,
    "table": 0.40
}

# Phase 2 feature flags
ENABLE_IMAGE_COMPARISON = True
ENABLE_LAYOUT_COMPARISON = True
ENABLE_METADATA_COMPARISON = True

# CLIP model for image embeddings (Phase 2)
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIMENSION = 512

# File constraints
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".pdf", ".docx"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Logging
LOG_LEVEL = "INFO"
