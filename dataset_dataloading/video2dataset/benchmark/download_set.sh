#!/bin/bash

video2dataset --url_list="video_sets/mp4.parquet" \
        --input_format="parquet" \
        --output_folder="dataset/mp4" \
        --output-format="webdataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --enable_wandb=False \
        --video_size=360 \
        --number_sample_per_shard=10 \
        --processes_count 10  \
