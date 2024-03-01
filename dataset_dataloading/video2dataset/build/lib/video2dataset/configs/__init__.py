"""video2dataset example configs"""
import os
from omegaconf import OmegaConf

configs_path = os.path.dirname(os.path.realpath(__file__))

CONFIGS = {
    "default": OmegaConf.load(os.path.join(configs_path, "default.yaml")),
    "downsample_ml": OmegaConf.load(os.path.join(configs_path, "downsample_ml.yaml")),
    "optical_flow": OmegaConf.load(os.path.join(configs_path, "optical_flow.yaml")),
    "caption": OmegaConf.load(os.path.join(configs_path, "caption.yaml")),
}
