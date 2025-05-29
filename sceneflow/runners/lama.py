import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from sceneflow.runners._factory import INPAINTERS
from sceneflow.utils.hub import download_model_weights_to_zoo

from ._helpers import ModelRunner

current_dir = Path(__file__).resolve().parent
third_party_path = current_dir.parent.parent / "third_party"
lama_path = third_party_path / "lama"

if str(lama_path) not in sys.path:
    sys.path.insert(0, str(lama_path))

from saicinpainting.evaluation.data import pad_tensor_to_modulo  # noqa: E402
from saicinpainting.evaluation.utils import move_to_device  # noqa: E402
from saicinpainting.training.trainers import load_checkpoint  # noqa: E402


class LamaRunner(ModelRunner):
    def _load_model(self):
        if None in (load_checkpoint, move_to_device, pad_tensor_to_modulo):
            raise ImportError(
                "LaMa inpainting module is not available. Make sure the 'lama' code is in 'third_party/lama'."
            )

        # Config
        config_path = lama_path / "configs" / "prediction" / "default.yaml"
        ckpt_path = download_model_weights_to_zoo(self.model_name)

        self.device = torch.device(self.device)

        predict_config = OmegaConf.load(config_path)

        # Training config
        predict_config.model.path = str(ckpt_path)
        train_config_path = os.path.join(predict_config.model.path, "config.yaml")

        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        checkpoint_path = os.path.join(predict_config.model.path, "models", predict_config.model.checkpoint)

        self._model = load_checkpoint(train_config, checkpoint_path, strict=False)
        self._model.to(self.device)
        self._model.freeze()
        self.out_key = predict_config.out_key or "inpainted"

    @torch.no_grad()
    def run(self, image: np.ndarray, mask: np.ndarray, **kwargs):
        if np.max(mask) <= 1:
            mask = (mask * 255).astype(np.uint8)

        image_tensor = torch.from_numpy(image).float().div(255.0)
        mask_tensor = torch.from_numpy(mask).float()

        batch = {"image": image_tensor.permute(2, 0, 1).unsqueeze(0), "mask": mask_tensor[None, None]}

        unpad_to_size = batch["image"].shape[2:]
        batch["image"] = pad_tensor_to_modulo(batch["image"], 8)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], 8)
        batch = move_to_device(batch, self.device)
        batch["mask"] = (batch["mask"] > 0).float()

        result = self._model(batch)[self.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        result = result[: unpad_to_size[0], : unpad_to_size[1]]
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return result


@INPAINTERS.register("big_lama")
def big_lama(name: str = "big_lama", device: str = "cuda") -> LamaRunner:
    return LamaRunner(name, device=device)
