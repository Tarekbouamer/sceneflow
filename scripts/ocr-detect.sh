INPUT_DIR="assets/images/subtitles"
OUTPUT_DIR="assets/results/subtitles"

sceneflow-ocr-detect --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --det-thd 0.5
