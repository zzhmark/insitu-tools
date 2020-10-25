#!/usr/bin/env bash

PROJECT_FOLDER=$(git rev-parse --show-toplevel)
GENES=$(ls "$PROJECT_FOLDER/data")
STAGE=7-8
IMAGE_IDS=(insitu67049 insitu30437 insitu53132 insitu18855 insitu53110 insitu67016 insitu64380 insitu71591)
OUTPUT_DIR="$PROJECT_FOLDER/examples/output/all-in-one"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi
i=0
for gene in $GENES; do
    insituTools findPatterns \
        --inputImage "$PROJECT_FOLDER/data/$gene/$STAGE/${IMAGE_IDS[$i]}.bmp" \
        --outputDirectory "$OUTPUT_DIR" \
        --downSampleFactor 9 \
        --noRotation --noRemoveBackground
done