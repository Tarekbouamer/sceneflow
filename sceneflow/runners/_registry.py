from typing import Any, Callable, Dict, List, Optional


class ModelRegistry:
    """
    A named registry for model runner instances or classes.
    Each model is responsible for its own loading and execution.
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Callable] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, meta: Optional[Dict[str, Any]] = None):
        """
        Decorator to register a model runner (class or instance).
        """

        def decorator(runner: Callable):
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in '{self.name}'")
            self._registry[name] = runner
            self._meta[name] = meta or {}
            return runner

        return decorator

    def get(self, name: str) -> Callable:
        """
        Retrieve the registered model runner.
        """
        if name not in self._registry:
            raise ValueError(f"'{name}' not found in registry '{self.name}'")
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry

    def unregister(self, name: str):
        if name in self._registry:
            del self._registry[name]
            self._meta.pop(name, None)
        else:
            raise ValueError(f"'{name}' is not registered in '{self.name}'")

    def list(self) -> Dict[str, Callable]:
        return dict(self._registry)

    def list_models(self) -> List[str]:
        return sorted(self._registry.keys())

    def summary(self) -> str:
        if not self._registry:
            return f"[{self.name}] (empty)"
        lines = [f"[{self.name}] {len(self._registry)} model(s):"]
        for name in self.list_models():
            lines.append(f"  • {name}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        model_list = ", ".join(self.list_models()) or "no models"
        return f"<ModelRegistry '{self.name}' with {len(self._registry)} model(s): {model_list}>"
