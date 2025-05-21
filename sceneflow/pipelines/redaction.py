from pathlib import Path
from typing import List, Optional

import torch

from sceneflow.core.camouflage import Camouflage
from sceneflow.core.mask_generator import MaskGenerator, load_detector, load_segmentor
from sceneflow.utils.draw import blend_detections, create_predictions, generate_static_scene_mask
from sceneflow.utils.io import (
    get_all_images,
    load_image,
    save_annotations,
    save_image,
    save_mask,
)
from sceneflow.utils.logger import logger
from sceneflow.utils.progress import get_progress


def redact(
    input_dir: str,
    output_dir: str,
    detector: str,
    segmentor: str,
    det_thd: float,
    allowed_classes: Optional[str],
    camouflage_method: str,
):
    # Paths
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    det_model = load_detector(detector)
    sam_predictor, _ = load_segmentor(segmentor, device=device)
    mask_gen = MaskGenerator(det_model, sam_predictor, device=device)
    camouflage = Camouflage(method=camouflage_method)

    logger.info(f"Detector  : {det_model.__class__.__name__}")
    logger.info(f"Segmentor : {type(sam_predictor).__name__}")
    logger.info(f"Camouflage : {camouflage_method}")

    allow: List[str | int] | None = None
    if allowed_classes:
        allow = [c.strip() for c in allowed_classes.split(",") if c.strip()]

    images = get_all_images(in_path)
    if not images:
        logger.warning("No images found.")
        return

    with get_progress() as progress:
        task = progress.add_task("Processing images...", total=len(images))

        for img_path in images:
            img = load_image(img_path)
            if img is None:
                logger.warning(f"Skipping invalid image: {img_path}")
                progress.advance(task)
                continue

            # Generate masks
            detections, masks, _ = mask_gen.generate(img, det_thd=det_thd, allowed_classes=allow)

            if not detections:
                logger.info(f"No detections for {img_path.name}")
                progress.advance(task)
                continue

            # Apply camouflage
            inpainted = camouflage.hide(img.copy(), masks)

            blended = blend_detections(img, detections)
            static_mask = generate_static_scene_mask(img, detections)

            rel = img_path.relative_to(in_path)
            base = out_path / rel
            base.parent.mkdir(parents=True, exist_ok=True)

            save_image(blended, base)
            save_image(inpainted, base.with_suffix(".inpainted.png"))
            save_mask(static_mask, base.with_suffix(".mask.png"))
            save_annotations(detections, base.with_suffix(".json"))

            progress.advance(task)

    logger.info(f"✔ Finished. Results saved to: {out_path}")
