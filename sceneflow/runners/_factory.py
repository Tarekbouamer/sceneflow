from ._helpers import ModelRunner
from ._registry import ModelRegistry

DETECTORS = ModelRegistry("detectors")
OVD_DETECTORS = ModelRegistry("ovd_detectors")
SEGMENTORS = ModelRegistry("segmentors")
TEXT_DETECTORS = ModelRegistry("text_detectors")
INPAINTERS = ModelRegistry("inpainters")


def load_detector(name: str, device: str = "cpu", kwargs: dict = {}) -> ModelRunner:
    return DETECTORS.get(name, device=device, **kwargs)


def load_ovd_detector(name: str, device: str = "cpu", kwargs: dict = {}) -> ModelRunner:
    return OVD_DETECTORS.get(name, device=device, **kwargs)


def load_segmentor(name: str, device: str = "cpu", kwargs: dict = {}) -> ModelRunner:
    return SEGMENTORS.get(name, device=device, **kwargs)


def load_text_detector(name: str, device: str = "cpu", kwargs: dict = {}) -> ModelRunner:
    return TEXT_DETECTORS.get(name, device=device, **kwargs)


def load_inpainter(name: str, device: str = "cpu", kwargs: dict = {}) -> ModelRunner:
    return INPAINTERS.get(name, device=device, **kwargs)
