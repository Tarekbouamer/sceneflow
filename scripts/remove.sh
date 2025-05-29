INPUT_DIR="assets/images/mots"
OUTPUT_DIR="assets/results/mots"

ALLOWED_CLASSES="people,person"

sceneflow-remove --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --prompt $ALLOWED_CLASSES \
    --det-thd 0.2
