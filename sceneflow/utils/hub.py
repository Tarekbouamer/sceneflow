from pathlib import Path
from typing import Union

import torch

from .logger import logger

MODEL_ZOO = {
    # RTDETR
    "rtdetr_l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt",
    "rtdetr_xl": "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt",
    # Sam
    "sam_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


ZOO_DIR = Path("zoo")


def download_model_weights_to_zoo(model_name: str) -> Union[Path, None]:
    model_name = model_name.lower()
    if model_name not in MODEL_ZOO:
        return None

    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    dest = ZOO_DIR / f"{model_name}.pt"
    url = MODEL_ZOO[model_name]

    if not dest.exists():
        logger.info(f"Downloading {model_name} from {url} → {dest}")
        try:
            torch.hub.download_url_to_file(url, dest)
            logger.success(f"Downloaded {model_name}.")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    return dest
