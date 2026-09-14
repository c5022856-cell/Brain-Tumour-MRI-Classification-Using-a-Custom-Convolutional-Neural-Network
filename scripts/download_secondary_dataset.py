from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

LOGGER = logging.getLogger("secondary_dataset_download")
DATASET_SLUG = "sartajbhuvaji/brain-tumor-classification-mri"


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


def configure_kaggle_credentials(project_root: Path) -> None:
    local_kaggle_dir = project_root / "developer-only" / "kaggle"
    local_kaggle_json = local_kaggle_dir / "kaggle.json"
    if local_kaggle_json.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(local_kaggle_dir)
        return
    raise FileNotFoundError(
        f"Kaggle credentials not found. Put kaggle.json in {local_kaggle_json}"
    )


def secondary_source_exists(source_dir: Path) -> bool:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            return True
    return False


def clean_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def download_archive(download_dir: Path) -> Path:
    archive_path = download_dir / "brain-tumor-classification-mri.zip"
    command = [
        sys.executable,
        "-m",
        "kaggle.cli",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(download_dir),
        "--force",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_lines: list[str] = []
    progress = tqdm(total=100, desc="Downloading secondary dataset", unit="%")
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
            "Kaggle secondary dataset download failed with exit code "
            f"{return_code}. Output:\n" + "\n".join(output_lines)
        )

    if archive_path.exists():
        return archive_path

    zip_files = sorted(download_dir.glob("*.zip"))
    if len(zip_files) != 1:
        raise FileNotFoundError(f"Expected one zip file in {download_dir}, found {len(zip_files)}")
    return zip_files[0]


def extract_archive(archive_path: Path, source_dir: Path) -> None:
    clean_directory_contents(source_dir)
    LOGGER.info("Extracting %s into %s", archive_path, source_dir)
    shutil.unpack_archive(str(archive_path), str(source_dir))


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parents[1]
    enforce_local_storage(project_root)
    configure_kaggle_credentials(project_root)

    download_dir = project_root / "data" / "raw" / "secondary" / "downloads"
    source_dir = project_root / "data" / "raw" / "secondary" / "source"
    download_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    if secondary_source_exists(source_dir):
        LOGGER.info("Secondary source dataset already exists in %s. Skipping download.", source_dir)
        return

    LOGGER.info(
        "Using secondary dataset %s because it is a public alternate MRI dataset with the same four classes "
        "(glioma, meningioma, pituitary, no tumor), which fits the proposal's cross-dataset validation goal.",
        DATASET_SLUG,
    )
    archive_path = download_archive(download_dir)
    extract_archive(archive_path, source_dir)
    LOGGER.info("Secondary dataset download completed successfully.")


if __name__ == "__main__":
    main()
