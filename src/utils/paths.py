from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root(root_value: str) -> Path:
    return Path(root_value).resolve()


def enforce_local_storage(root: Path, cache_dir: Path) -> None:
    root = root.resolve()
    cache_dir = cache_dir.resolve()

    os.environ["BRAIN_TUMOR_PROJECT_ROOT"] = str(root)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface" / "transformers")


def project_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"Path escapes project root: {path}")
    return path

