#!/usr/bin/env bash

PROJECT_FOLDER=$(git rev-parse --show-toplevel)
GENES=$(ls "$PROJECT_FOLDER/data")
OUTPUT_DIR="$PROJECT_FOLDER/examples/output/scores"

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

i=0
for gene in GENES; do
    INPUT_PREFIXES[$i]="$PROJECT_FOLDER/examples/output/all-in-one/$gene";
    i=$(expr $i + 1);
done

# pairwise
insituTools score \
	$INPUT_PREFIXES[*]
	--outputTablePath "$OUTPUT_DIR/pairwise.csv"
	--noFlipping

# reference-based
insituTools score \
	$INPUT_PREFIXES[*]
	--outputTablePath "$OUTPUT_DIR/reference-based.csv"
	--reference 0,2,4
	--noFlipping