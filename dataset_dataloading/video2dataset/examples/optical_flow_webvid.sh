#!/bin/bash

video2dataset --url_list="dataset/{00000..00004}.tar" \
        --input_format="webdataset" \
        --output-format="webdataset" \
	--output_folder="dataset_optical_flow" \
	--stage "optical_flow" \
	--encode_formats '{"optical_flow": "npy"}' \
	--config "optical_flow" \
