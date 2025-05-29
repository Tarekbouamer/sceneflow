import re
import zipfile
from pathlib import Path
from typing import Union

import gdown
import torch

from sceneflow.utils.logger import logger

ZOO_DIR = Path("zoo")

MODEL_ZOO = {
    # RT-DETR
    "rtdetr_l": {
        "type": "url",
        "path": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt",
    },
    "rtdetr_xl": {
        "type": "url",
        "path": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
    },
    # SAM
    "sam_h": {
        "type": "url",
        "path": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "sam_l": {
        "type": "url",
        "path": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "sam_b": {
        "type": "url",
        "path": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    # Lama
    "big_lama": {
        "type": "file",
        "path": "zoo/big-lama.zip",  # FIXME: replace with gDrive or uil if needed
    },
}


def unzip_model_zip_file(zip_path: Union[str, Path], keep: bool = False) -> Union[Path, None]:
    """Unzips a model zip file to the zoo directory."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        logger.error(f"Zip file does not exist: {zip_path}")
        return None

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(ZOO_DIR)

    extracted_dir = ZOO_DIR / zip_path.stem
    logger.success(f"Extracted {zip_path.name} to {extracted_dir}")

    if not keep:
        zip_path.unlink()

    return extracted_dir.resolve()


def download_model_weights_to_zoo(model_name: str) -> Union[Path, None]:
    """Downloads the specified model weights to the zoo directory."""
    model_name = model_name.lower()
    entry = MODEL_ZOO.get(model_name)
    if not entry:
        logger.error(f"Model '{model_name}' not found in MODEL_ZOO.")
        return None

    model_type = entry["type"]
    model_path = entry["path"]

    if model_type == "file":
        local_path = Path(model_path)

        if local_path.suffix == ".zip":
            unzip_model_zip_file(local_path)
            # FIXME: quick fix
            return local_path.with_suffix("").resolve()

    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(model_path).name.split("?")[0]
    dest = ZOO_DIR / filename

    if dest.exists():
        if dest.suffix == ".zip":
            return unzip_model_zip_file(dest)
        return dest

    logger.info(f"Downloading {model_name} from {model_path} → {dest}")

    try:
        if model_type == "url":
            torch.hub.download_url_to_file(model_path, str(dest))
        elif model_type == "gdrive":
            if gdown is None:
                raise RuntimeError("gdown is required for Google Drive downloads.")
            file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", model_path)
            if not file_id_match:
                raise ValueError(f"Could not extract file ID from Google Drive URL: {model_path}")
            file_id = file_id_match.group(1)
            gdown.download(id=file_id, output=str(dest), quiet=False)
        else:
            raise ValueError(f"Unknown model type '{model_type}' for model '{model_name}'")

        logger.success(f"Downloaded {model_name} to {dest}")

        if dest.suffix == ".zip":
            return unzip_model_zip_file(dest)

        return dest

    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
        return None
