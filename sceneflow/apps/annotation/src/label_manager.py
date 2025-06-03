from typing import Dict, List, Optional

import streamlit as st

from sceneflow.apps.annotation.src.annotation_store import AnnotationStore


class LabelManager:
    """Manage label definitions and sync with annotation store."""

    def __init__(self) -> None:
        self._annotation_store: Optional[AnnotationStore] = None
        if "labels" not in st.session_state:
            st.session_state["labels"] = []
        else:
            self._normalize()

    def _normalize(self) -> None:
        """Ensure all labels have full structure (name, id, color)."""
        normalized: List[Dict[str, str]] = []
        for item in st.session_state["labels"]:
            if isinstance(item, str):
                normalized.append({"name": item, "id": item, "color": "#CCCCCC"})
            elif isinstance(item, dict):
                normalized.append(item)
        st.session_state["labels"] = normalized

    def set_annotation_store(self, store: AnnotationStore) -> None:
        """Attach annotation store to push label updates to XML."""
        self._annotation_store = store
        # Also load labels from file if present and not yet in session
        if not st.session_state["labels"]:
            st.session_state["labels"] = self._annotation_store.get_labels()

    def _sync_labels_to_store(self) -> None:
        """Push current labels to annotation store and save to XML."""
        if self._annotation_store:
            self._annotation_store.set_labels(self.all_labels())
            self._annotation_store.save()

    def all_labels(self) -> List[Dict[str, str]]:
        """Return all current label definitions."""
        return st.session_state["labels"]

    def add_label(self, name: str, id_: str, color: str) -> bool:
        """Add a new label if not duplicate, and sync to store."""
        for lbl in st.session_state["labels"]:
            if lbl["name"] == name:
                st.warning(f"Label name '{name}' already exists.")
                return False
            if lbl["id"] == id_:
                st.warning(f"Label ID '{id_}' already exists.")
                return False
            if lbl["color"] == color:
                st.warning(f"Label color '{color}' already exists.")
                return False

        st.session_state["labels"].append({"name": name, "id": id_, "color": color})
        st.success(f"Added label: {name} (ID={id_})")
        self._sync_labels_to_store()
        return True

    def remove_label(self, index: int) -> None:
        """Remove label at given index and sync to store."""
        st.session_state["labels"].pop(index)
        self._sync_labels_to_store()
        st.rerun()
