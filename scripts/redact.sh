INPUT_DIR="images"
OUTPUT_DIR="results"

ALLOWED_CLASSES="person,car,bicycle,cat,dog,window,door,tree,river,road,sky,grass,building,boat"

sceneflow-redact --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --allowed-classes $ALLOWED_CLASSES \
    --resize 1920 1080 \
    --det-thd 0.35
