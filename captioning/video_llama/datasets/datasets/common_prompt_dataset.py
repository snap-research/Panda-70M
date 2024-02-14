"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
import json
import numpy as np

random.seed(0)

class CommonPromptDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_json):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.annotation = json.load(open(ann_json))
        self.vis_root = vis_root

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            # process video
            video_name = self.annotation[index]["video"]
            video_path = os.path.join(self.vis_root, video_name)
            try:
                video = self.vis_processor(video_path)
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            # process caption
            caption = self.annotation[index]["caption"]
            if isinstance(caption, list):
                caption = random.sample(caption, 1)[0]
            assert isinstance(caption, str)
            caption = self.text_processor(caption)

            # if isinstance(caption, str):
            #     caption = self.text_processor(caption)
            # elif isinstance(caption, list):
            #     caption = [self.text_processor(c) for c in caption]
            # else:
            #     raise NotImplementedError            

            # process prompt
            prompt = "Please faithfully summarize the following video in one sentence."

            if video is None or video.size()!=torch.Size([3,self.vis_processor.n_frms,self.vis_processor.image_size,self.vis_processor.image_size]):
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format

        return {
            "image": video,
            "prompt": prompt,
            "caption": caption,
            "type":'video',
            "video_path": video_path,
        }

    def __len__(self):
        return len(self.annotation)
        