# 🐼 Panda-70M: Long Video Splitting
The section includes the code to split a long video into multiple semantics-consistent short clips.

## Video Splitting and Quick Demo
### Setup Repository and Enviroment
```
git clone https://github.com/snap-research/Panda-70M.git
cd Panda-70M/splitting

# create a conda environment
conda create --name panda70m_splitting python=3.8.16 -y
conda activate panda70m_splitting
pip install -r requirements.txt

# install ffmpeg
apt install ffmpeg
```

### Step 1: Shot Boundary Detection
```
python cutscene_detect.py --video-list video_list.txt --output-json-file cutscene_frame_idx.json
```
The code will process the videos listed in `video_list.txt` and detect the cutscenes of the videos. The results (index of the video frames) will be saved in `cutscene_frame_idx.json` with the format:
```
{
    "video1.mp4": [[0, 149], [149, 298], [298, 479], [479, 628], [628, 777], [777, 926], [926, 1094], [1094, 1243], [1243, 1428], [1428, 1577], [1577, 1726], [1726, 1875], [1875, 2024], [2024, 2173], [2173, 2322], [2322, 2397], [2397, 2546], [2546, 2695], [2695, 2829], [2829, 2978], [2978, 3127], [3127, 3204], [3204, 3353], [3353, 3520], [3520, 3597]],
    "video2.mp4": [[0, 149], [149, 298], [298, 447], [447, 596], [596, 711], [711, 860], [860, 1009], [1009, 1158], [1158, 1307], [1307, 1456], [1456, 1579], [1579, 1728], [1728, 1877], [1877, 2026], [2026, 2175], [2175, 2324], [2324, 2478], [2478, 2627], [2627, 2776], [2776, 2925], [2925, 3074], [3074, 3223], [3223, 3372], [3372, 3521], [3521, 3597]]
}
```
- **[Note]** To handle videos with complex transititions that cannot be detected as cutscenes (e.g., fade-in or fade-out) or deal with unedited footages, we recursively cut the first 5 seconds as a new clip if the video clip is longer than 7 seconds.

### Step 2: Stitching 
```
python event_stitching.py --video-list video_list.txt --cutscene-frameidx cutscene_frame_idx.json --output-json-file event_timecode.json
```
The code will process the videos listed in `video_list.txt` and stitch the semantically-similar adjacent clips into a longer scene. The results (timecodes of the events) will be saved in `event_timecode.json` with the format:
```
{
    "video1.mp4": [["0:00:00.000", "0:00:15.982"], ["0:00:15.982", "0:00:36.503"], ["0:00:36.503", "0:00:57.590"], ["0:00:57.590", "0:01:19.979"], ["0:01:19.979", "0:01:34.394"], ["0:01:34.394", "0:01:46.906"], ["0:01:46.906", "0:02:00.019"]],
    "video2.mp4": [["0:00:00.000", "0:00:23.723"], ["0:00:23.723", "0:00:52.685"], ["0:00:52.685", "0:01:22.682"], ["0:01:22.682", "0:02:00.019"]]
}
```
- **[Note]** We make several changes of the parameters for better splitting results. If you want to use the same parameters as we collect Panda-70M, you can run [line 200](https://github.com/snap-research/Panda-70M/blob/70226bd6d8ce3fc35b994b2d13273b57d5469da5/splitting/event_stitching.py#L200) and comment out [line 199](https://github.com/snap-research/Panda-70M/blob/70226bd6d8ce3fc35b994b2d13273b57d5469da5/splitting/event_stitching.py#L199).
### Step 3: Video Splitting
```
python video_splitting.py --video-list video_list.txt --event-timecode event_timecode.json --output-folder outputs
```
The code will split the videos listed in the `video_list.txt` and output the video clips to `outputs` folder. Here are some output examples:
<table class="center">
    <tr style="line-height: 0">
        <td width=25% style="border: none; text-align: center">video1.0.mp4</td>
        <td width=25% style="border: none; text-align: center">video1.1.mp4</td>
        <td width=25% style="border: none; text-align: center">video1.2.mp4</td>
        <td width=25% style="border: none; text-align: center">video1.3.mp4</td>
    </tr>
    <tr>
        <td width=25% style="border: none"><img src="assets/video1.0.gif" style="width:100%"></td>
        <td width=25% style="border: none"><img src="assets/video1.1.gif" style="width:100%"></td>
        <td width=25% style="border: none"><img src="assets/video1.2.gif" style="width:100%"></td>
        <td width=25% style="border: none"><img src="assets/video1.3.gif" style="width:100%"></td>
    </tr>
</table>

<sup>**We will remove the video samples from our dataset / Github / project webpage as long as you need it. Please contact tsaishienchen at gmail dot com for the request.</sup>

## Acknowledgements
The code for video splitting is built upon [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) and [ImageBind](https://github.com/facebookresearch/ImageBind).
Thanks for sharing the great work!
