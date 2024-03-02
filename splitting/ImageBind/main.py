import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import numpy as np
from tqdm import tqdm
import argparse, json, re, data, time, os


def repl_func(match: re.Match):
    return " ".join(match.group().split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageBind")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--input-json-file", type=str, required=True)
    parser.add_argument("--output-json-file", type=str, default=None)
    args = parser.parse_args() 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    f = open(args.input_json_file)
    video_captions = json.load(f)

    for video in tqdm(video_captions.keys()):
        video_path = os.path.join(args.data_path, video)
        captions = video_captions[video]["captions"]

        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(list(captions.values()), device),
            ModalityType.VISION: data.load_and_transform_video_data([video_path], device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        matching_scores = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T
        matching_scores = list(matching_scores.to('cpu').tolist())[0]
        
        video_captions[video]["imagebind"] = [list(captions.keys())[np.array(matching_scores).argmax()], \
                                              matching_scores]
        
    if args.output_json_file:
        json_object = json.dumps(video_captions, indent = 4)
        json_object = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, json_object)
        with open(args.output_json_file, "w") as f:
            f.write(json_object)
    else:
        for key, item in video_captions.items():
            print(key)
            print(item)
