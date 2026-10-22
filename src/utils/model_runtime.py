"""Reuse downloaded models across comparisons and retrieval in one process."""
from functools import lru_cache
import os
import config


def _load_cached(factory, name, **kwargs):
    # Once a model is present, local analysis must not wait on Hub network probes.
    try:
        return factory(name, local_files_only=True, **kwargs)
    except OSError:
        return factory(name, **kwargs)


@lru_cache(maxsize=2)
def get_text_model():
    import torch
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "4"))))
    from sentence_transformers import SentenceTransformer
    return _load_cached(SentenceTransformer, config.TEXT_EMBEDDING_MODEL, cache_folder=str(config.MODEL_CACHE_DIR))


@lru_cache(maxsize=2)
def get_clip_runtime(model_name=config.CLIP_MODEL):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_cached(CLIPModel.from_pretrained, model_name, cache_dir=str(config.MODEL_CACHE_DIR)).to(device)
    model.eval()
    processor = _load_cached(CLIPProcessor.from_pretrained, model_name, cache_dir=str(config.MODEL_CACHE_DIR))
    return model, processor, device


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return _load_cached(CrossEncoder, config.RERANKER_MODEL, cache_folder=str(config.MODEL_CACHE_DIR))
