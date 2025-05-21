
# SceneFlow

SceneFlow is a lightweight **scene‑parsing toolkit**.  
It provides ready‑made pipelines that chain **object detection, instance segmentation, tracking, and in‑painting**—and its modular layout lets you swap models or add new workflows as your project grows.

```bash
pip install sceneflow
```

## Built‑in pipeline

| Command            | Stages                                      | Artefacts per image                              |
|--------------------|---------------------------------------------|--------------------------------------------------|
| `sceneflow-redact` | detect → mask → camouflage (in‑painting)    | preview RGB · inpainted RGB · bg‑mask PNG · JSON |

More pipelines (mask‑export, background isolation, MOT tracking…) are planned.

## Quick start

```bash
sceneflow-redact --input-dir  images/ --output-dir out/ --detector   rtdetr_l --segmentor  sam_l
```

### Extending

1. **Models** → add loader in `sceneflow/core`  
2. **Workflows** → drop a file in `sceneflow/pipelines`  
3. **CLI** → register the Click command in `sceneflow/cli`

The clear separation of *core*, *pipelines*, and *CLI* keeps SceneFlow easy to grow.
