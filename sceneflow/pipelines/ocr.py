import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import torch

from sceneflow.core.ocr_processor import OCRProcessor
from sceneflow.utils.draw import blend_detections
from sceneflow.utils.io import (
    get_all_images,
    load_image,
    save_image,
)
from sceneflow.utils.logger import logger
from sceneflow.utils.progress import get_progress


def detect_text_boxes(
    input_dir: Path,
    output_dir: Path,
    text_detector: str,
    det_thd: float = 0.0,
    resize: Optional[Tuple[int, int]] = None,
):
    # Paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    # Load OCRProcessor
    processor = OCRProcessor.from_pretrained([text_detector], device=device)

    logger.info(f"Using text detector: {text_detector}")
    logger.info(f"Detection threshold: {det_thd}")
    logger.info(f"Resize images to: {resize}")

    # Stats
    per_class = Counter()
    n_processed = 0
    t_start = time.perf_counter()

    images_paths = get_all_images(input_dir)
    if len(images_paths) == 0:
        logger.warning("No images found.")
        return

    with get_progress() as progress:
        task = progress.add_task("Processing images", total=len(images_paths))

        for img_path in images_paths:
            img, img_bgr, original_size, scale = load_image(img_path, resize=resize)

            if img is None:
                logger.warning(f"Skipping invalid image: {img_path}")
                progress.advance(task)
                continue

            # Run OCR
            detections = processor.process(img, conf=det_thd, scale=scale)

            for d in detections:
                per_class[d.class_name] += 1

            n_processed += 1

            # Save paths
            relative_path = img_path.relative_to(input_dir)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save visual overlay
            if detections:
                blended = blend_detections(img_bgr, detections)
                save_image(blended, save_path.with_suffix(".detected.png"))

            # Save detections as JSON
            text_data = [d.to_json() for d in detections]
            with open(save_path.with_suffix(".json"), "w") as f:
                json.dump(text_data, f, indent=2)

            progress.advance(task)

    # Summary
    elapsed = time.perf_counter() - t_start
    summary = {
        "total_images": len(images_paths),
        "processed_images": n_processed,
        "total_text_boxes": int(sum(per_class.values())),
        "avg_time_per_image_sec": round(elapsed / max(1, n_processed), 3),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✔ Finished. Results saved to: {output_dir}")
