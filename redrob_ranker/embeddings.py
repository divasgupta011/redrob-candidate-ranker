"""Loader for the pre-computed embedding artifact used by the semantic/hybrid rankers.

``precompute.py`` (Step 9) embeds the pool offline and writes:
  artifacts/candidate_embeddings.npz   (arrays: 'ids' [str], 'vectors' [float32 N x D])
  artifacts/jd_vector.npy              (float32 D)

The ranking step then loads these with numpy alone -- no torch, no network -- keeping
it inside the 5-minute CPU budget. If the artifact is absent, this returns None and
the hybrid ranker degrades to structured-only, so the rank step never hard-fails.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_precomputed(artifacts_dir) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    d = Path(artifacts_dir)
    emb_path = d / "candidate_embeddings.npz"
    jd_path = d / "jd_vector.npy"
    if not emb_path.exists() or not jd_path.exists():
        return None
    data = np.load(emb_path, allow_pickle=True)
    ids = data["ids"]
    vectors = data["vectors"].astype(np.float32)
    jd_vec = np.load(jd_path).astype(np.float32)
    lookup = {str(cid): vectors[i] for i, cid in enumerate(ids)}
    return jd_vec, lookup
