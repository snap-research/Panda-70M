import os
import logging
import warnings
import yaml

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.webvid_datasets import WebvidDataset
from video_llama.datasets.datasets.hdvila_datasets import HdvilaDataset
from video_llama.datasets.datasets.common_prompt_dataset import CommonPromptDataset


@registry.register_builder("webvid")
class WebvidBuilder(BaseDatasetBuilder):

    DATASET_CONFIG_DICT = {"default": "configs/datasets/webvid/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        build_info = self.config.build_info

        if "train" in build_info.type:
            datasets["train"] = WebvidDataset(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                vis_root=build_info.videos_dir,
                ann_root=build_info.anno_dir,
                train_eval_split="train"
            )
        
        if "eval" in build_info.type:
            datasets["eval"] = WebvidDataset(
                vis_processor=self.vis_processors["eval"],
                text_processor=self.text_processors["eval"],
                vis_root=build_info.videos_dir,
                ann_root=build_info.anno_dir,
                train_eval_split="eval"
            )
        
        return datasets


@registry.register_builder("msrvtt")
class MSRVTTBuilder(BaseDatasetBuilder):

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        build_info = self.config.build_info

        if "train" in build_info.type:
            datasets["train"] = CommonPromptDataset(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[0]
            )
        
        if "eval" in build_info.type:
            datasets["eval"] = CommonPromptDataset(
                vis_processor=self.vis_processors["eval"],
                text_processor=self.text_processors["eval"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[1]
            )
        
        return datasets


@registry.register_builder("msvd")
class MSVDBuilder(BaseDatasetBuilder):

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msvd/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        build_info = self.config.build_info

        if "train" in build_info.type:
            datasets["train"] = CommonPromptDataset(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[0],
            )
        
        if "eval" in build_info.type:
            datasets["eval"] = CommonPromptDataset(
                vis_processor=self.vis_processors["eval"],
                text_processor=self.text_processors["eval"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[1]
            )
        
        return datasets


@registry.register_builder("didemo")
class DiDeMoBuilder(BaseDatasetBuilder):

    DATASET_CONFIG_DICT = {"default": "configs/datasets/didemo/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        build_info = self.config.build_info

        if "train" in build_info.type:
            datasets["train"] = CommonPromptDataset(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[0],
            )
        
        if "eval" in build_info.type:
            datasets["eval"] = CommonPromptDataset(
                vis_processor=self.vis_processors["eval"],
                text_processor=self.text_processors["eval"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[1]
            )
        
        return datasets


@registry.register_builder("hdvila")
class HdvilaBuilder(BaseDatasetBuilder):

    DATASET_CONFIG_DICT = {"default": "configs/datasets/hdvila/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        build_info = self.config.build_info

        if "train" in build_info.type:
            datasets["train"] = HdvilaDataset(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[0],
                p_sub=build_info.p_sub[0],
                p_meta=build_info.p_meta[0],
                max_char_len=build_info.max_char_len
            )
        
        if "eval" in build_info.type:
            datasets["eval"] = HdvilaDataset(
                vis_processor=self.vis_processors["eval"],
                text_processor=self.text_processors["eval"],
                vis_root=build_info.videos_dir,
                ann_json=build_info.anno_path[1],
                p_sub=build_info.p_sub[1],
                p_meta=build_info.p_meta[1],
                max_char_len=build_info.max_char_len
            )
        
        return datasets
