import glob
import argparse
import torch
import json
import os
from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors.video_processor import load_video
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--cfg-path", default="eval_configs/panda70M_eval.yaml", help="path to configuration file.")
    parser.add_argument("--video-list", required=True, help="list of input videos.")
    parser.add_argument("--output-json", default=None, help="output json file. Leave none to print out the results.")
    parser.add_argument("--prompt-list", default=None, help="list of correponding input prompts. Leave none if no prompt input.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to("cuda")
    model.eval()

    vis_processor_cfg = DotDict({"name":"alpro_video_eval", "n_frms":8, "image_size":224})
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor_cfg = DotDict({"name":"blip_caption", "max_words":100})
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)

    batch_size = 16
    
    videos = open(args.video_list, "r").read().splitlines()    
    if args.prompt_list:
        prompts = open(args.prompt_list, "r").read().split("\n\n")
    
    results = {}
    for i in tqdm(range(0, len(videos), batch_size)):
        video_batch = []
        video_path_batch = []
        prompt_batch = []
        
        for j in range(i, min(i+batch_size, len(videos))):
            try:
                video_path = videos[j]
                video = load_video(video_path=video_path, n_frms=8, sampling ="uniform")
                video = vis_processor.transform(video)
                assert video.shape == torch.Size([3, 8, 224, 224])
            except Exception as e:
                print(e)
                continue

            video_batch.append(video)
            video_path_batch.append(video_path.split('/')[-1])
            prompt_batch.append(prompts[j] if args.prompt_list else "Please faithfully summarize the following video in one sentence.")
            
        video_batch = torch.stack(video_batch).to("cuda")
        outputs = model.inference(video_batch, prompt_batch)
        
        for video_path, output in zip(video_path_batch, outputs):
            output = output.capitalize()+"."
            if args.output_json:
                results[video_path] = output
            else:
                print("====="*20)
                print("[Input video]", video_path)
                print("[Input prompt]")
                print(prompt_batch[j-i])
                print("[Output caption]", output)

    if args.output_json:
        results = json.dumps(results, indent = 4)
        with open(args.output_json, "w") as f:
            f.write(results)
