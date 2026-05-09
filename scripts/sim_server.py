import logging
import os
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# use $SCRATCH is defined (for NERSC users); otherwise, use ~/.cache
CACHE_DIR = os.path.join(
    os.environ.get("SCRATCH", os.path.expanduser("~/.cache/mt-jailbench")),
    "sim",
)
MODEL_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"
HOST = "0.0.0.0"
PORT = 8000

app = FastAPI()


@app.on_event("startup")
def load_model():
    logger.info("Starting similarity server. Model cache location: %s", CACHE_DIR)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Missing local model dependencies. Install with: "
            "uv sync --extra local-model"
        ) from e

    device = os.environ.get("SIM_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model: %s on device=%s", MODEL_NAME, device)
    t0 = time.time()

    app.state.device = device
    app.state.model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        cache_folder=CACHE_DIR,
    )

    logger.info("Model loaded in %.2f seconds", time.time() - t0)


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class EmbeddingRequest(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": app.state.device,
    }


@app.post("/embed")
def embed(req: EmbeddingRequest):
    embeddings = app.state.model.encode(
        req.texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return {"embeddings": embeddings.tolist()}


@app.post("/similarity")
def similarity(req: SimilarityRequest):
    import numpy as np

    embs = app.state.model.encode(
        [req.text1, req.text2],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    score = float(np.dot(embs[0], embs[1]))
    return {"similarity": score}


def main():
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()