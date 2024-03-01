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


class HdvilaDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_json, p_sub, p_meta, max_char_len):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.annotation = json.load(open(ann_json))
        self.vis_root = vis_root
        self.p_sub = p_sub
        self.p_meta = p_meta
        self.max_char_len = max_char_len

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
            caption = self.text_processor(self.annotation[index]["caption"])

            # process prompt
            prompt = ""
            sub_path = video_path.replace(".mp4", ".fine_text.json")
            if os.path.isfile(sub_path) and np.random.uniform() < self.p_sub:
                d = json.load(open(sub_path))
                subtitle = []
                for sentence in d["intervals"]:
                    sentence = sentence["text"].strip()
                    if len(sentence) > 0:
                        subtitle.append(sentence)
                subtitle = " ".join(subtitle).strip().capitalize()
                subtitle = subtitle[:self.max_char_len]
                prompt += "\nTranscription: %s"%subtitle

            meta_path = os.path.join(self.vis_root, video_name.split(".")[0]) + ".metadata.json"
            if os.path.isfile(meta_path) and np.random.uniform() < self.p_meta:
                d = json.load(open(meta_path))
                title = d["title"]
                title = title[:self.max_char_len]
                description = d["description"]
                if description:
                    description = description.strip().replace("\n", " ")
                    description = description[:self.max_char_len]

                metadata = []
                if title:
                    metadata.append(title)
                if description:
                    metadata.append(description)
                if len(metadata) > 0:
                    prompt += "\nMetadata: %s"%(str(metadata))

            if prompt:
                prompt = "Some information about a video you will get:" + prompt + "\nPlease look at the video and faithfully summarize it in one sentence."
            else:
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

    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result
        