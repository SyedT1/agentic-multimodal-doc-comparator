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

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding configuration
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # MiniLM output dimension

# Chunking parameters
TEXT_CHUNK_SIZE = 512  # tokens
TEXT_CHUNK_OVERLAP = 50  # tokens

# Similarity parameters
TOP_K_MATCHES = 10  # Number of similar chunks to retrieve

# Modality weights (Phase 1: text + tables only)
# These weights must sum to 1.0
MODALITY_WEIGHTS = {
    "text": 0.60,
    "table": 0.40
}

# File constraints
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".pdf", ".docx"]

# Future: LLM API keys (Phase 2)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Logging
LOG_LEVEL = "INFO"
