import json
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from sceneflow.core.camouflage import Camouflage
from sceneflow.core.mask_generator import MaskGenerator
from sceneflow.utils.draw import blend_detections, generate_static_scene_mask
from sceneflow.utils.io import (
    get_all_images,
    load_image,
    save_image,
    save_mask,
)
from sceneflow.utils.logger import logger
from sceneflow.utils.progress import get_progress


def redact(
    input_dir: str,
    output_dir: str,
    detectors: str,
    ovd_detectors: str,
    segmentor: str,
    nms_iou: float,
    det_thd: float,
    allowed_classes: Optional[str],
    camouflage_method: str,
    resize: Optional[Tuple[int, int]] = None,
):
    # Paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    mask_gen = MaskGenerator.from_pretrained(detectors, ovd_detectors, segmentor, device=device)
    camouflage = Camouflage(method=camouflage_method)

    logger.info(f"Detector  : {mask_gen.__class__.__name__}")
    logger.info(f"Segmentor : {type(camouflage).__name__}")
    logger.info(f"Camouflage : {camouflage_method}")

    allow: List[str | int] | None = None
    if allowed_classes:
        allow = [c.strip() for c in allowed_classes.split(",") if c.strip()]

    logger.info(f"Allowed classes: {allow}")
    logger.info(f"Resize images to: {resize}")
    logger.info(f"NMS IoU threshold: {nms_iou}")
    logger.info(f"Detection threshold: {det_thd}")

    images_paths = get_all_images(input_dir)
    if len(images_paths) == 0:
        logger.warning("No images found.")
        return

    # Stats
    per_class = Counter()
    n_processed = 0
    n_skipped = 0
    t_start = time.perf_counter()

    # Process images
    with get_progress() as progress:
        task = progress.add_task("Processing images", total=len(images_paths))

        for img_path in images_paths:
            # Load image
            img, img_bgr, original_size, scale = load_image(img_path, resize=resize)

            if img is None:
                logger.warning(f"Skipping invalid image: {img_path}")
                progress.advance(task)
                continue

            # Generate masks
            detections, masks, _ = mask_gen.generate(
                img, conf=det_thd, allowed_classes=allow, scale=scale, original_size=original_size, nms_iou=nms_iou
            )

            # stats update
            for d in detections:
                per_class[d["class_name"]] += 1

            n_processed += 1

            # Save paths
            relative_path = img_path.relative_to(input_dir)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Apply camouflage
            if len(masks) > 0:
                inpainted = camouflage.hide(img_bgr.copy(), masks)
                save_image(inpainted, save_path.with_suffix(".camouflaged.png"))

            if len(detections) > 0:
                blended = blend_detections(img_bgr, detections)
                save_image(blended, save_path.with_suffix(".blended.png"))

            # Save
            static_mask = generate_static_scene_mask(img_bgr, detections)

            save_mask(static_mask, save_path.parent / (save_path.name + ".png"))

            progress.advance(task)

    # Write summary
    elapsed = time.perf_counter() - t_start
    summary = {
        "total_images": len(images_paths),
        "processed_images": n_processed,
        "skipped_images": n_skipped,
        "total_detections_removed": int(sum(per_class.values())),
        "per_class_counts": dict(per_class),
        "avg_time_per_image_sec": round(elapsed / max(1, n_processed), 3),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✔ Finished. Results saved to: {output_dir}")
