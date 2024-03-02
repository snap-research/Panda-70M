import os, json, argparse
from datetime import datetime
import subprocess
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stitching")
    parser.add_argument("--video-list", type=str, required=True)
    parser.add_argument("--event-timecode", type=str, required=True)
    parser.add_argument("--output-folder", type=str, default="outputs")
    args = parser.parse_args()

    f = open(args.video_list, "r")
    video_paths = f.read().splitlines()
    f = open(args.event_timecode)
    video_timecodes = json.load(f)

    os.makedirs(args.output_folder, exist_ok=True)

    for video_path in tqdm(video_paths):
        video_name = video_path.split("/")[-1].split('.')[0]
        timecodes = video_timecodes[video_path.split("/")[-1]]

        for i, timecode in enumerate(timecodes): 
            start_time = datetime.strptime(timecode[0], '%H:%M:%S.%f')
            end_time = datetime.strptime(timecode[1], '%H:%M:%S.%f')
            video_duration = (end_time - start_time).total_seconds()
            os.system("ffmpeg -hide_banner -loglevel panic -ss %s -t %.3f -i %s %s"%(timecode[0], video_duration, video_path, os.path.join(args.output_folder, video_name+".%i.mp4"%i)))
