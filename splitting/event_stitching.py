import torch
from torchvision import transforms
from PIL import Image
import cv2
from multiprocessing import Pool
from collections import defaultdict
import os, sys, re, json, argparse
from datetime import datetime, timedelta
from pytz import timezone
from tqdm import tqdm
import numpy as np

sys.path.append('ImageBind')
from models import imagebind_model
from models.imagebind_model import ModalityType


def read_videoframe(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    res, frame = cap.read()
    if res:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224,224), interpolation = cv2.INTER_LINEAR)
    else:
        frame = np.zeros((224,224,3), dtype=np.uint8)
    return frame, res


def transfer_timecode(frameidx, fps):
    timecode = []
    for (start_idx, end_idx) in frameidx:
        s = str(timedelta(seconds=start_idx/fps, microseconds=1))[:-3]
        e = str(timedelta(seconds=end_idx/fps, microseconds=1))[:-3]
        timecode.append([s, e])
    return timecode


def extract_cutscene_feature(video_path, cutscenes):
    features = torch.empty((0,1024))
    res = []
    num_parallel_cutscene = 128
    for i in range(0, len(cutscenes), num_parallel_cutscene):
        cutscenes_sub = cutscenes[i:i+num_parallel_cutscene]
        frame_idx = []
        for cutscene in cutscenes_sub:
            start_frame_idx = int(0.95*cutscene[0] + 0.05*(cutscene[1]-1))
            end_frame_idx = int(0.05*cutscene[0] + 0.95*(cutscene[1]-1))
            frame_idx.extend([(video_path, start_frame_idx), (video_path, end_frame_idx)])

        with Pool(8) as p:
            results = p.starmap(read_videoframe, frame_idx)
        frames = [image_transform(Image.fromarray(i[0])) for i in results]
        res.extend([i[1] for i in results])

        frames = torch.stack(frames, dim=0)
        with torch.no_grad():
            batch_features = model({ModalityType.VISION: frames.to(device)})[ModalityType.VISION]
        features = torch.vstack((features, batch_features.detach().cpu()))

    return features, res


def verify_cutscene(cutscenes, cutscene_feature, cutscene_status, transition_threshold=0.8):
    cutscenes_new = []
    cutscene_feature_new = []
    for i, cutscene in enumerate(cutscenes):
        start_frame_fet, end_frame_fet = cutscene_feature[2*i], cutscene_feature[2*i+1]
        start_frame_res, end_frame_res = cutscene_status[2*i], cutscene_status[2*i+1]
        diff = (start_frame_fet - end_frame_fet).pow(2).sum().sqrt()

        # Remove condition 1: start_frame or end_frame cannot be loaded
        if not (start_frame_res and end_frame_res):
            continue
        # Remove condition 2: cutscene include scene transition effect
        if diff > transition_threshold:
            continue

        cutscenes_new.append(cutscene)
        cutscene_feature_new.append([start_frame_fet, end_frame_fet])
    return cutscenes_new, cutscene_feature_new


def cutscene_stitching(cutscenes, cutscene_feature, eventcut_threshold=0.6):
    assert len(cutscenes) == len(cutscene_feature)
    num_cutscenes = len(cutscenes)

    events = []
    event_feature = []
    for i in range(num_cutscenes):
        # The first cutscene or the cutscene is discontinuous from the previous one => start a new event
        if i == 0 or cutscenes[i][0] != events[-1][-1]:
            events.append(cutscenes[i])
            event_feature.append(cutscene_feature[i])
            continue

        diff = (event_feature[-1][-1] - cutscene_feature[i][0]).pow(2).sum().sqrt()
        # The difference between the cutscene and the previous one is large => start a new event
        if diff > eventcut_threshold:
            events.append(cutscenes[i])
            event_feature.append(cutscene_feature[i])
        # Otherwise => extend the previous event
        else:
            events[-1].extend(cutscenes[i])
            event_feature[-1].extend(cutscene_feature[i])

    if len(events[-1]) == 0:
        events.pop(-1)
        event_feature.pop(-1)

    return events, event_feature


def verify_event(events, event_feature, fps, min_event_len=1.5, max_event_len=60, redundant_event_threshold=0.4, trim_begin_last_percent=0.1, still_event_threshold=0.1): # add remove no change event
    assert len(events) == len(event_feature)
    num_events = len(events)
    
    events_final = []
    event_feature_final = torch.empty((0,1024))
    
    min_event_len = min_event_len*fps
    max_event_len = max_event_len*fps

    for i in range(num_events):
        assert len(events[i]) == len(event_feature[i])
        # Remove condition 1: shorter than min_event_len
        if (events[i][-1] - events[i][0]) < min_event_len:
            continue
            
        # Remove condition 2: within-event difference is too small
        diff = (event_feature[i][0] - event_feature[i][-1]).pow(2).sum().sqrt()
        if diff < still_event_threshold:
            continue
        
        avg_feature = torch.stack(event_feature[i]).mean(axis=0)
        # Remove condition 3: too similar to the previous events
        diff = (event_feature_final - avg_feature).pow(2).sum(axis=1).sqrt()
        if torch.any(diff < redundant_event_threshold):
            continue

        # Trim the event if it is too long
        events[i][-1] = events[i][0] + min(int(max_event_len), (events[i][-1]-events[i][0]))
        
        trim_len = int(trim_begin_last_percent*(events[i][-1]-events[i][0]))
        events_final.append([events[i][0]+trim_len, events[i][-1]-trim_len])
        event_feature_final = torch.vstack((event_feature_final, avg_feature))
        
    return events_final, event_feature_final


def write_json_file(data, output_file):
    data = json.dumps(data, indent = 4)
    def repl_func(match: re.Match):
        return " ".join(match.group().split())
    data = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, data)
    data = re.sub(r'\[\s+', '[', data)
    data = re.sub(r'],\s+\[', '], [', data)
    data = re.sub(r'\s+\]', ']', data)
    with open(output_file, "w") as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stitching")
    parser.add_argument("--video-list", type=str, required=True)
    parser.add_argument("--cutscene-frameidx", type=str, required=True)
    parser.add_argument("--output-json-file", type=str, default="event_timecode.json")
    args = parser.parse_args()

    device = "cuda"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    f = open(args.video_list, "r")
    video_paths = f.read().splitlines()
    f = open(args.cutscene_frameidx)
    video_cutscenes = json.load(f)

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    video_events = {}
    for video_path in tqdm(video_paths):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cutscene = video_cutscenes[video_path.split("/")[-1]]

        cutscene_raw_feature, cutscene_raw_status = extract_cutscene_feature(video_path, cutscene)
        cutscenes, cutscene_feature = verify_cutscene(cutscene, cutscene_raw_feature, cutscene_raw_status, transition_threshold=1.)
        events_raw, event_feature_raw = cutscene_stitching(cutscenes, cutscene_feature, eventcut_threshold=0.6)
        events, event_feature = verify_event(events_raw, event_feature_raw, fps, min_event_len=2.0, max_event_len=1200, redundant_event_threshold=0.0, trim_begin_last_percent=0.0, still_event_threshold=0.15)
        # events, event_feature = verify_event(events_raw, event_feature_raw, fps, min_event_len=2.5, max_event_len=60, redundant_event_threshold=0.3, trim_begin_last_percent=0.1, still_event_threshold=0.15)
        video_events[video_path.split("/")[-1]] = transfer_timecode(events, fps)
       
    write_json_file(video_events, args.output_json_file)
