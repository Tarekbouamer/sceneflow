from ._helpers import ModelRunner
from ._registry import ModelRegistry

DETECTORS = ModelRegistry("detectors")
OVD_DETECTORS = ModelRegistry("ovd_detectors")
SEGMENTORS = ModelRegistry("segmentors")


def load_detector(name: str) -> ModelRunner:
    return DETECTORS.get(name)()


def load_ovd_detector(name: str) -> ModelRunner:
    return OVD_DETECTORS.get(name)()


def load_segmentor(name: str) -> ModelRunner:
    return SEGMENTORS.get(name)()
