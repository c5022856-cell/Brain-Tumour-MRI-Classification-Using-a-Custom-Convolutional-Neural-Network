from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


LOGGER = logging.getLogger("dataset_download")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def enforce_local_storage(project_root: Path) -> None:
    cache_dir = project_root / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["BRAIN_TUMOR_PROJECT_ROOT"] = str(project_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface" / "transformers")


def configure_kaggle_credentials(project_root: Path) -> None:
    local_kaggle_dir = project_root / "developer-only" / "kaggle"
    local_kaggle_json = local_kaggle_dir / "kaggle.json"

    if local_kaggle_json.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(local_kaggle_dir)
        LOGGER.info("Using Kaggle credentials from %s", local_kaggle_json)
        return

    existing_kaggle_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if existing_kaggle_dir:
        LOGGER.info("Using existing KAGGLE_CONFIG_DIR=%s", existing_kaggle_dir)
        return

    raise FileNotFoundError(
        "Kaggle credentials not found. Put kaggle.json in "
        f"{local_kaggle_json}"
    )


def run_download(project_root: Path) -> None:
    download_dir = project_root / "data" / "raw" / "downloads"
    archive_path = download_dir / "brain-tumor-mri-dataset.zip"
    primary_dir = project_root / "data" / "raw" / "primary"

    download_dir.mkdir(parents=True, exist_ok=True)
    primary_dir.mkdir(parents=True, exist_ok=True)

    if dataset_exists(primary_dir):
        LOGGER.info("Existing normalized dataset found in %s. Skipping download.", primary_dir)
        return

    command = [
        sys.executable,
        "-m",
        "kaggle.cli",
        "datasets",
        "download",
        "-d",
        "masoudnickparvar/brain-tumor-mri-dataset",
        "-p",
        str(download_dir),
        "--force",
    ]

    LOGGER.info("Downloading dataset archive into %s", download_dir)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    progress = tqdm(total=100, desc="Downloading dataset", unit="%")
    last_percent = 0

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        LOGGER.info("%s", line)
        match = re.search(r"(\d+)%", line)
        if match:
            percent = int(match.group(1))
            if percent > last_percent:
                progress.update(percent - last_percent)
                last_percent = percent

    return_code = process.wait()
    if return_code == 0 and last_percent < 100:
        progress.update(100 - last_percent)
    progress.close()

    if return_code != 0:
        raise RuntimeError(
            "Kaggle download failed with exit code "
            f"{return_code}. Output:\n" + "\n".join(output_lines)
        )

    if not archive_path.exists():
        matches = sorted(download_dir.glob("*.zip"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one zip file in {download_dir}, found {len(matches)}"
            )
        archive_path = matches[0]

    extract_dir = project_root / "data" / "raw" / "downloads" / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Extracting %s into %s", archive_path, extract_dir)
    shutil.unpack_archive(str(archive_path), str(extract_dir))

    normalize_dataset(extract_dir, primary_dir)


def dataset_exists(primary_dir: Path) -> bool:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = [path for path in primary_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        return False

    for class_dir in class_dirs:
        if any(path.is_file() and path.suffix.lower() in valid_extensions for path in class_dir.rglob("*")):
            return True
    return False


def normalize_dataset(extract_dir: Path, primary_dir: Path) -> None:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        path for path in extract_dir.rglob("*") if path.is_file() and path.suffix.lower() in valid_extensions
    ]
    if not image_paths:
        raise FileNotFoundError(f"No image files found after extraction in {extract_dir}")

    for path in primary_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)

    copied = 0
    for image_path in tqdm(image_paths, desc="Normalizing dataset layout"):
        label = infer_label(image_path)
        if label is None:
            continue
        destination_dir = primary_dir / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{image_path.parent.name}_{image_path.name}"
        shutil.copy2(image_path, destination_path)
        copied += 1

    if copied == 0:
        raise RuntimeError("Failed to normalize dataset into class folders.")
    LOGGER.info("Normalized %s images into %s", copied, primary_dir)


def infer_label(image_path: Path) -> str | None:
    ignored = {"training", "testing", "train", "test", "val", "validation"}
    parts = [part.lower() for part in image_path.parts]
    if len(parts) < 2:
        return None

    parent_name = image_path.parent.name
    if parent_name.lower() not in ignored:
        return parent_name

    for part in reversed(image_path.parts[:-1]):
        if part.lower() not in ignored:
            return part
    return None


def verify_dataset(project_root: Path) -> None:
    dataset_dir = project_root / "data" / "raw" / "primary"
    image_paths = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            image_paths.append(path)

    class_counts: dict[str, int] = {}
    for image_path in tqdm(image_paths, desc="Verifying downloaded images"):
        label = image_path.parent.name
        class_counts[label] = class_counts.get(label, 0) + 1

    LOGGER.info("Verified %s images across %s classes", len(image_paths), len(class_counts))
    for class_name, count in sorted(class_counts.items()):
        LOGGER.info("Class %s: %s images", class_name, count)


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    enforce_local_storage(project_root)
    configure_kaggle_credentials(project_root)
    run_download(project_root)
    verify_dataset(project_root)
    LOGGER.info("Dataset download completed successfully.")


if __name__ == "__main__":
    main()
