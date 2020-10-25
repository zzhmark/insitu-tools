#!/usr/bin/env bash

PROJECT_FOLDER=$(git rev-parse --show-toplevel)
GENES=$(ls "$PROJECT_FOLDER/data")
STAGE=7-8
IMAGE_IDS=(insitu67049 insitu30437 insitu53132 insitu18855 insitu53110 insitu67016 insitu64380 insitu71591)
OUTPUT_DIR="$PROJECT_FOLDER/examples/output/step-by-step"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

i=0
for gene in $GENES; do
    IMAGE_PATH="$PROJECT_FOLDER/data/$gene/$STAGE/${IMAGE_IDS[$i]}.bmp";
    insituTools extract \
	    --inputImage "$IMAGE_PATH" \
	    --outputMask "$OUTPUT_DIR/${IMAGE_IDS[$i]}_mask.bmp";
    insituTools register \
	    --inputImage "$IMAGE_PATH" \
    	--inputMask "$OUTPUT_DIR/${IMAGE_IDS[$i]}_mask.bmp" \
	    --outputImage "$OUTPUT_DIR/${IMAGE_IDS[$i]}_registered.bmp" \
        --outputMask "$OUTPUT_DIR/${IMAGE_IDS[$i]}_mask_registered.bmp" \
        --downSampleFactor 9 \
        --noRotation;
    insituTools globalGMM \
        --inputImage "$OUTPUT_DIR/${IMAGE_IDS[$i]}_registered.bmp" \
        --inputMask "$OUTPUT_DIR/${IMAGE_IDS[$i]}_mask_registered.bmp" \
        --outputLabel "$OUTPUT_DIR/${IMAGE_IDS[$i]}_label_global.bmp" \
        --outputLevels "$OUTPUT_DIR/${IMAGE_IDS[$i]}_levels_global.txt" \
        --outputImage "$OUTPUT_DIR/${IMAGE_IDS[$i]}_global.bmp";
    insituTools localGMM \
        --inputImage "$OUTPUT_DIR/${IMAGE_IDS[$i]}_registered.bmp" \
        --inputLabel "$OUTPUT_DIR/${IMAGE_IDS[$i]}_label_global.bmp" \
        --inputLevels "$OUTPUT_DIR/${IMAGE_IDS[$i]}_levels_global.txt" \
        --outputLabel "$OUTPUT_DIR/${IMAGE_IDS[$i]}_label_local.bmp" \
        --outputLevels "$OUTPUT_DIR/${IMAGE_IDS[$i]}_levels_local.txt" \
        --outputImage "$OUTPUT_DIR/${IMAGE_IDS[$i]}_local.bmp";
    i=$(expr $i + 1);
done

