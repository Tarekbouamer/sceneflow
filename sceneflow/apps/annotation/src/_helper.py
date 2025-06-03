import os

import streamlit as st

from sceneflow.apps.annotation.src.image_manager import ImageManager


@st.cache_data(show_spinner=False)
def get_all_files_cached(img_dir: str):
    """Return sorted list of image files in directory."""
    allow_exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(allow_exts)]
    return sorted(files)


@st.cache_data(show_spinner=False)
def get_annotation_files_cached(img_dir: str):
    """Return list of .xml annotation files in directory."""
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(".xml")]
    return sorted(files)


@st.cache_data(show_spinner=False)
def load_and_resize_image(img_path: str):
    """Return resized image and resized rects."""
    im = ImageManager(img_path)
    return im.resizing_img(), im.get_resized_rects()
