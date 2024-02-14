#!/bin/bash

video2dataset --url_list="dataset/{00000..00004}.tar" \
        --input_format="webdataset" \
        --output-format="webdataset" \
	--output_folder="dataset_downsampled" \
	--stage "subset" \
	--config "downsample_ml" \
