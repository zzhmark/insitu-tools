#!/usr/bin/env bash

#PROJECT_FOLDER=$(git rev-parse --show-toplevel)
PROJECT_FOLDER=.
IMAGE_IDS="insitu67049 insitu30437 insitu53132 insitu18855 insitu53110 insitu67016 insitu64380 insitu71591"
OUTPUT_DIR="$PROJECT_FOLDER/examples/output/scores"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

i=0
for id in $IMAGE_IDS; do
    INPUT_PREFIXES[$i]='"'"$PROJECT_FOLDER/examples/output/all-in-one/$id"'"';
    i=$(expr $i + 1);
done

# pairwise
insituTools score \
	${INPUT_PREFIXES[*]} \
	--outputTablePath "$OUTPUT_DIR/pairwise.csv" \
	--noFlipping

# reference-based
insituTools score \
	${INPUT_PREFIXES[*]} \
	--outputTablePath "$OUTPUT_DIR/reference-based.csv" \
	--reference 0,2,4 \
	--noFlipping