from pathlib import Path
import streamlit as st
from src import st_img_label
from streamlit_autorefresh import st_autorefresh

from sceneflow.apps.annotation.src._helper import (
    get_all_files_cached,
    load_and_resize_image,
)
from sceneflow.apps.annotation.src.annotation_store import AnnotationStore
from sceneflow.apps.annotation.src.image_manager import ImageManager
from sceneflow.apps.annotation.src.label_manager import LabelManager


class ImageAnnotationApp:
    def __init__(self):
        self.label_mgr = LabelManager()
        self.img_dir = None
        self.files = []
        self.total_images = 0
        self.current_file = None
        self.img_path = None
        self.annotation_store = None

        if "dirty" not in st.session_state:
            st.session_state["dirty"] = False
        if "image_index" not in st.session_state:
            st.session_state["image_index"] = 0

    def mark_dirty(self):
        st.session_state["dirty"] = True

    def refresh(self):
        st.cache_data.clear()
        st.session_state["files"] = get_all_files_cached(self.img_dir)
        st.session_state["image_index"] = 0
        self.files = st.session_state["files"]
        self.total_images = len(self.files)

    def save_current(self):
        if not self.current_file:
            return

        index = st.session_state["image_index"]
        img = ImageManager(self.img_path)
        _, resized_rects = load_and_resize_image(self.img_path)

        original_rects = []
        label_id_map = {lbl["name"]: lbl["id"] for lbl in self.label_mgr.all_labels()}

        for i, r in enumerate(resized_rects):
            key = f"label_{index}_{i}"
            label = st.session_state.get(key, "")
            original_rects.append({
                "label": label,
                "id": label_id_map.get(label, ""),
                "left": int(r["left"] * img._resized_ratio_w),
                "top": int(r["top"] * img._resized_ratio_h),
                "width": int(r["width"] * img._resized_ratio_w),
                "height": int(r["height"] * img._resized_ratio_h),
            })

        self.annotation_store.update(self.current_file, original_rects)
        self.annotation_store.save()
        st.session_state["dirty"] = False
        st.toast(f"Saved: {self.current_file}")

    def autosave(self):
        if st.session_state["dirty"]:
            self.save_current()
            st.toast("Auto-saved annotations")

    def next_image(self):
        if st.session_state["image_index"] < self.total_images - 1:
            st.session_state["image_index"] += 1

    def previous_image(self):
        if st.session_state["image_index"] > 0:
            st.session_state["image_index"] -= 1

    def next_to_annotate(self):
        current_index = st.session_state["image_index"]
        for i in range(current_index + 1, self.total_images):
            if self.files[i] not in self.annotation_store.annotations:
                st.session_state["image_index"] = i
                return
        for i in range(0, current_index):
            if self.files[i] not in self.annotation_store.annotations:
                st.session_state["image_index"] = i
                return
        st.warning("All images are annotated.")
        self.next_image()

    def run(self):
        if st_autorefresh(interval=60000, limit=None, key="autosave_timer"):
            self.autosave()

        self.img_dir = st.sidebar.text_input("Enter image directory path:", "")
        if not self.img_dir or not Path(self.img_dir).is_dir():
            st.warning("Please enter a valid directory path.")
            return

        self.annotation_store = AnnotationStore(self.img_dir)
        self.label_mgr.set_annotation_store(self.annotation_store)

        if "files" not in st.session_state:
            st.session_state["files"] = get_all_files_cached(self.img_dir)
        self.files = st.session_state["files"]
        self.total_images = len(self.files)

        index = st.session_state["image_index"]
        self.current_file = self.files[index]
        self.img_path = str(Path(self.img_dir) / self.current_file)

        # Sidebar: navigation
        st.sidebar.selectbox(
            "Select Image",
            self.files,
            index=index,
            key="file",
            on_change=lambda: st.session_state.update({"image_index": self.files.index(st.session_state["file"])}),
        )

        col1, col2 = st.sidebar.columns(2)
        col1.button("Previous", on_click=self.previous_image, use_container_width=True)
        col2.button("Next", on_click=self.next_image, use_container_width=True)

        col3, col4 = st.sidebar.columns(2)
        col3.button("Next to Annotate", on_click=self.next_to_annotate, use_container_width=True)
        col4.button("Refresh", on_click=self.refresh, use_container_width=True)

        # Sidebar: Label manager
        with st.sidebar.expander("Manage Labels", expanded=True):
            col_name, col_id, col_color = st.columns([3, 1, 1])
            new_name = col_name.text_input("Name:", key="new_label_name")
            new_id = col_id.text_input("ID:", key="new_label_id")
            new_color = col_color.color_picker("Color:", "#ff0000", key="new_label_color")
            if st.button("Add label"):
                if new_name and new_id:
                    self.label_mgr.add_label(new_name.strip(), new_id.strip(), new_color)
                else:
                    st.warning("Both name and ID required.")

            labels = self.label_mgr.all_labels()
            if labels:
                options = [lbl["name"] for lbl in labels]
                selected = st.selectbox("Labels:", options, key="current_labels_dropdown")
                if st.button("Remove Label", key="remove_label_button"):
                    self.label_mgr.remove_label(options.index(selected))

        # Sidebar: save + stats
        st.sidebar.button("Save Progress", on_click=self.save_current, use_container_width=True)
        st.sidebar.markdown("**Statistics**")
        st.sidebar.write("Labels:", len(self.label_mgr.all_labels()))
        st.sidebar.write("Total images:", self.total_images)
        st.sidebar.write("Saved:", len(self.annotation_store.annotations))
        st.sidebar.write("Remaining:", self.total_images - len(self.annotation_store.annotations))

        # Main: annotation UI
        st.markdown(f"**Image {index+1}/{self.total_images}**  \n**File:** `{self.current_file}`")

        new_index = st.slider("Navigate images", 0, self.total_images - 1, index)
        if new_index != index:
            st.session_state["image_index"] = new_index
            self.current_file = self.files[new_index]
            self.img_path = str(Path(self.img_dir) / self.current_file)

        resized_img, resized_rects = load_and_resize_image(self.img_path)
        rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

        if rects:
            st.button("Save & Next", on_click=lambda: (self.save_current(), self.next_to_annotate()))
            label_names = [lbl["name"] for lbl in self.label_mgr.all_labels()]
            previews = ImageManager(self.img_path).init_annotation(rects)
            for i, (thumb, prev_label) in enumerate(previews):
                thumb.thumbnail((200, 200))
                colA, colB = st.columns(2)
                colA.image(thumb)
                default = label_names.index(prev_label) if prev_label in label_names else 0
                colB.selectbox("Label", label_names, index=default, key=f"label_{index}_{i}", on_change=self.mark_dirty)


if __name__ == "__main__":
    app = ImageAnnotationApp()
    app.run()
