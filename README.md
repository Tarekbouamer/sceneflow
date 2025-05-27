# SceneFlow

SceneFlow is a lightweight **scene‑parsing toolkit**.  
It provides ready‑made pipelines that chain **object detection, instance segmentation, tracking, and in‑painting**—and its modular layout lets you swap models or add new workflows as your project grows.

---

## Built‑in Pipelines

| Command                | Stages                                    | Artefacts per image                                        |
|------------------------|-------------------------------------------|------------------------------------------------------------|
| `sceneflow redact`     | detect → mask → camouflage (in‑painting)  | preview RGB · inpainted RGB · bg‑mask PNG · JSON           |
| `sceneflow ocr-detect` | detect → recognize text (OCR)             | JSON with text boxes, scores, and recognized text          |

More pipelines (mask export, background isolation, MOT tracking…) are planned.

---

## Quick Start

### 🔒 Redaction Pipeline

```bash
sceneflow redact --input-dir images/ --output-dir out/ --detector rtdetr_l --segmentor sam_l
```

### 🔤 OCR Detection

```bash
sceneflow ocr-detect --input-dir images/ --output-dir ocr_out/ --text-detector mmocr_dbnet_abinet
```

---

## Extending SceneFlow

### 🔧 Add Models

- Create a loader in `sceneflow/core`

### 📦 Add Pipelines

- Drop a pipeline script into `sceneflow/pipelines`

### 🖥️ Add CLI Commands

- Register your command in `sceneflow/cli/__init__.py` using `cli.add_command(...)`

SceneFlow’s separation of *core*, *pipelines*, and *CLI* keeps it clean and scalable.

---

## Building SceneFlow

We use **[Pixi](https://prefix.dev/pixi)** (recommended) and **pip** to manage development environments and dependencies.

```bash
# Clone the repo
git clone https://github.com/Tarekbouamer/sceneflow.git
cd sceneflow

# Create the environment and install dependencies
pixi install          # installs Python, Torch, CUDA, etc.
pixi run post-install # installs MMCV/MMOCR from source

# Open a shell in the dev environment
pixi shell

# Install SceneFlow locally (editable mode)
pip install -e .
```

You’re now ready to run, modify, and contribute to SceneFlow.

---
