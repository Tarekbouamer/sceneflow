import json
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

from sceneflow.core.mask_generator import MaskGenerator
from sceneflow.core.remover import Remover
from sceneflow.utils.io import get_all_images, load_image, save_image
from sceneflow.utils.logger import logger
from sceneflow.utils.progress import get_progress


def remove_objects_with_prompts(
    input_dir: str,
    output_dir: str,
    ovd_detector: str,
    segmentor: str,
    inpainter: str,
    prompt: str,
    det_thd: float = 0.25,
    resize: Optional[Tuple[int, int]] = None,
    nms_iou: float = 0.5,
):
    # Paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    logger.info(f"Prompt classes: {prompt}")
    logger.info(f"Using OVD detector: {ovd_detector}")
    logger.info(f"Segmentor: {segmentor}")
    logger.info(f"Inpainter: {inpainter}")
    logger.info(f"Detection threshold: {det_thd}")
    logger.info(f"Resize images to: {resize}")

    prompt = [p.strip() for p in prompt.split(",") if p.strip()]
    if not prompt:
        logger.warning("No valid prompt classes provided.")
        return

    # Init models
    mask_gen = MaskGenerator.from_pretrained(
        detectors=[],
        ovd_detectors=[ovd_detector],
        segmentor=segmentor,
        device=device,
    )
    remover = Remover(inpainter=inpainter, device=device)

    summary = {
        "total_images": 0,
        "processed_images": 0,
        "removed_objects": 0,
        "avg_time_per_image_sec": 0.0,
    }

    image_paths = get_all_images(input_dir)
    if not image_paths:
        logger.warning("No images found.")
        return

    t_start = time.perf_counter()

    with get_progress() as progress:
        task = progress.add_task("Removing objects", total=len(image_paths))

        for img_path in image_paths:
            img = load_image(img_path, resize=resize)[0]
            if img is None:
                logger.warning(f"Skipping invalid image: {img_path}")
                progress.advance(task)
                continue

            # Detect and segment
            _, masks, _ = mask_gen.generate(
                img.copy(),
                conf=det_thd,
                prompt=prompt,
                nms_iou=nms_iou,
            )
            if not masks.any():
                logger.info(f"No objects found for: {img_path.name}")
                progress.advance(task)
                continue

            # Inpaint
            inpainted = remover.remove(img.copy(), masks)

            # Save
            relative_path = img_path.relative_to(input_dir)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_image(inpainted, save_path.with_suffix(".inpainted.png"))

            summary["processed_images"] += 1
            summary["removed_objects"] += len(masks)

            progress.advance(task)

    elapsed = time.perf_counter() - t_start
    summary["total_images"] = len(image_paths)
    summary["avg_time_per_image_sec"] = round(elapsed / max(1, summary["processed_images"]), 3)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✔ Done. Results saved to {output_dir}")
