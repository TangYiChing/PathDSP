#!/bin/bash

# arg 1: output directory to download model-specific data
# If run via container, it needs to be downloaded under the csa_data folder

OUTPUT_DIR=$1

# Check if the data is already downloaded
if [ -f "$OUTPUT_DIR/.downloaded" ]; then 
  echo "Data present, skipping download"
# Download data if no other download is in progress
elif [ ! -f "$OUTPUT_DIR/.downloading_author_data" ]; then
  touch "$OUTPUT_DIR/.downloading_author_data"
  # Download files
  # Unzip files
  wget -P $OUTPUT_DIR https://zenodo.org/record/6093818/files/MSigdb.zip
  wget -P $OUTPUT_DIR https://zenodo.org/record/6093818/files/raw_data.zip
  wget -P $OUTPUT_DIR https://zenodo.org/record/6093818/files/STRING.zip
  unzip -d $OUTPUT_DIR $OUTPUT_DIR/MSigdb.zip
  unzip -d $OUTPUT_DIR $OUTPUT_DIR/raw_data.zip
  unzip -d $OUTPUT_DIR $OUTPUT_DIR/STRING.zip
  touch "$OUTPUT_DIR/.downloaded"
  rm "$OUTPUT_DIR/.downloading_author_data"
else
  # Wait for other download to finish
  iteration=0
  echo "Waiting for external download"
  while [ -f "$OUTPUT_DIR/.downloading_author_data" ]; do
    iteration=$((iteration + 1))
    if [ "$iteration" -gt 10 ]; then
      # Download takes too long, exit and warn user
      echo "Check output directory, download still in progress after $iteration minutes."
      exit 1
    fi
    sleep 60
  done
fi
