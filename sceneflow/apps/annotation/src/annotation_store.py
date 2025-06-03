import json
from pathlib import Path
from typing import Dict, List, Union


class AnnotationStore:
    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)
        self.filepath = self.directory / "annotations.json"
        self.annotations: Dict[str, List[Dict]] = {}
        self.labels: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not self.filepath.exists():
            return

        with self.filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            self.labels = data.get("labels", [])
            self.annotations = data.get("annotations", {})

    def update(self, filename: str, rects: List[Dict]) -> None:
        self.annotations[filename] = []
        label_id_map = {lbl["name"]: lbl["id"] for lbl in self.labels}

        for box in rects:
            label = box.get("label", "")
            obj = {
                "label": label,
                "id": label_id_map.get(label, ""),
                "bbox": [
                    int(box["left"]),
                    int(box["top"]),
                    int(box["width"]),
                    int(box["height"]),
                ],
            }
            self.annotations[filename].append(obj)
            
        

    def delete(self, filename: str) -> None:
        if filename in self.annotations:
            del self.annotations[filename]

    def get(self, filename: str) -> List[Dict]:
        return self.annotations.get(filename, [])

    def set_labels(self, labels: List[Dict]) -> None:
        self.labels = labels

    def get_labels(self) -> List[Dict]:
        return self.labels

    def save(self) -> None:
        out = {
            "labels": self.labels,
            "annotations": self.annotations,
        }
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
