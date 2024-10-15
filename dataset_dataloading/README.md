# 🐼 Panda-70M: Dataset Dataloading
The section includes the csv files listing the data samples in Panda-70M and the code to download the videos.

**[Note] Please use the video2dataset tool from this repository to download the dataset, as the video2dataset from [the official repository](https://github.com/iejMac/video2dataset) cannot work with our csv format.**

## Data Splitting and Download Link
  | Split           | Download | # Source Videos | # Samples | Video Duration | Storage Space |
  |-----------------|----------|-----------------|-----------|----------------|---------------|
  | Training (full) | [link](https://drive.google.com/file/d/1pbh8W3qgst9CD7nlPhsH9wmUSWjQlGdW/view?usp=sharing) (2.73 GB) | 3,779,763 | 70,723,513 | 167 khrs  | ~36 TB  |
  | Training (10M)  | [link](https://drive.google.com/file/d/1LLOFeYw9nZzjT5aA1Wj4oGi5yHUzwSk5/view?usp=sharing) (504 MB)  | 3,755,240 | 10,473,922 | 37.0 khrs | ~8.0 TB |
  | Training (2M)   | [link](https://drive.google.com/file/d/1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB/view?usp=sharing) (118 MB)  | 800,000   | 2,400,000  | 7.56 khrs | ~1.6 TB |
  | Validation      | [link](https://drive.google.com/file/d/1uHR5iXS3Sftzw6AwEhyZ9RefipNzBAzt/view?usp=sharing) (1.2 MB)  | 2,000     | 6,000      | 18.5 hrs  | ~4.0 GB |
  | Testing         | [link](https://drive.google.com/file/d/1BZ9L-157Au1TwmkwlJV8nZQvSRLIiFhq/view?usp=sharing) (1.2 MB)  | 2,000     | 6,000      | 18.5 hrs  | ~4.0 GB |
- Validation and testing set are collected from 2,000 source videos which do not appear in any training set to avoid testing information leakage. For each source video, we randomly sample 3 clips.
- Training set (10M) is the high-quality subset of training set (full). In the subset, we only sample at most 3 clips from a source video to increase diversity and the video-caption matching scores are all larger than 0.43 to guarantee a better caption quality.
- Training set (2M) is randomly sampled from training set (10M) and include 3 clips for each source video.
- **[Note 1]** The training csv files are too large and are compressed into zip files. Please `unzip` to get the csv files.
- **[Note 2]** We will remove the video samples from our dataset as long as you need it. Please contact tsaishienchen at gmail dot com for the request.
 
## Download Dataset
### Setup Repository and Enviroment
```
git clone https://github.com/snap-research/Panda-70M.git
cd Panda-70M/dataset_dataloading/video2dataset
pip install -e .
cd ..
```
### Download Dataset
Download the csv files and change `<csv_file>` and `<output_folder>` arguments to download corresponding data.
```
video2dataset --url_list="<csv_file>" \
              --url_col="url" \
              --caption_col="caption" \
              --clip_col="timestamp" \
              --output_folder="<output_folder>" \
              --save_additional_columns="[matching_score,desirable_filtering,shot_boundary_detection]" \
              --config="video2dataset/video2dataset/configs/panda70m.yaml"
```
### Known Issues
<table class="center">
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center"><b>Error Message</td>
    <td width=50% style="border: none; text-align: center"><b>Solution</td>
  </tr>
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center"><pre>pyarrow.lib.ArrowTypeError: Expected bytes, got<br>a 'list' object</pre></td>
    <td width=50% style="border: none; text-align: center">Your ffmpeg and ffmpeg-python version is out-of-date. Update them by pip or conda. Please refer <a href="https://github.com/kkroening/ffmpeg-python/issues/174">this issue</a> for more details.</td>
  </tr>
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center"><pre>HTTP Error 403: Forbidden</pre></td>
    <td width=50% style="border: none; text-align: center">Your IP got blocked. Use proxy for downloading. Please refer <a href="https://github.com/yt-dlp/yt-dlp/issues/8785">this issue</a> for more details.</td>
  </tr>
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center"><pre>HTTP Error 429: Too Many Requests</pre></td>
    <td width=50% style="border: none; text-align: center">Your download requests reach a limit. Slow down the download speed by reducing processes_count and thread_count in the <a href="./video2dataset/video2dataset/configs/panda_70M.yaml">config</a> file. Please refer <a href="https://github.com/iejMac/video2dataset/issues/267">this issue</a> for more details.</td>
  </tr>
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center"><pre>YouTube said: ERROR - Precondition check failed</pre></td>
    <td width=50% style="border: none; text-align: center">Your yt-dlp version is out-of-date and need to install a nightly version. Please refer <a href="https://github.com/yt-dlp/yt-dlp/issues/9316">this issue</a> for more details.</td>
  </tr>
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center">In the json file:<pre>"status": "failed_to_download" & "error_message":<br>"[Errno 2] No such file or directory: '/tmp/...'"</pre></td>
    <td width=50% style="border: none; text-align: center">The YouTube video has been set to private or removed. Please skip this sample.</td>
  </tr>
  <tr style="line-height: 0">
    <td width=50% style="border: none; text-align: center"><pre>YouTube: Skipping player responses from android clients<br>(got player responses for video ... instead of ...)</pre></td>
    <td width=50% style="border: none; text-align: center">The latest version of yt-dlp will solve this issue. Please refer <a href="https://github.com/yt-dlp/yt-dlp/issues/9554">this issue</a> for more details.</td>
  </tr>
</table>

### Dataset Format
The code will download and store the data with the format:
```
output-folder
 ├── 00000 {shardID}
 |     ├── 0000000_00000.mp4 {shardID + videoID _ clipID}
 |     ├── 0000000_00000.txt
 |     ├── 0000000_00000.json
 |     ├── 0000000_00001.mp4
 |     ├── 0000000_00001.txt
 |     ├── 0000000_00001.json
 |     └── ...
 |     ├── 0000099_00004.mp4
 |     ├── 0000099_00004.txt
 |     ├── 0000099_00004.json
 ├── 00001
 |     ├── 0000100_00000.mp4
 |     ├── 0000100_00000.txt
 |     ├── 0000100_00000.json
 │     ...
 ...
```
- Each data comes with 3 files: `.mp4` (video), `.txt` (caption), `.json` (meta information)
- Meta information includes:
  - Caption
  - Matching score: confidence score of each video-caption pair
  - **[New🔥]** Desirable filtering: whether a video is a suitable training sample for a video generation model. There are six categories of filtering results: `desirable`, `0_low_desirable_score`, `1_still_foreground_image`, `2_tiny_camera_movement`, `3_screen_in_screen`, `4_computer_screen_recording` (see [here](https://github.com/snap-research/Panda-70M) for examples in each category).
  - **[New🔥]** Shot boundary detection: a list of intervals representing continuous shots within a video (predicted by [TransNetV2](https://github.com/soCzech/TransNetV2)). If the list contains only one interval, it indicates that the video does not include any shot boundaries.
  - Other metadata: video title, description, categories, subtitles, to name but a few.
- **[Note 1]** The dataset is unshuffled and the clips from a same long video would be stored into a shard. Please manually shuffle them if needed.
- **[Note 2]** The videos are resized into 360 px height. You can change `download_size` in the [config](./video2dataset/video2dataset/configs/panda70m.yaml) file to get different video resolutions.
- **[Note 3]** The videos are downloaded with audio by default. You can change `download_audio` in the [config](./video2dataset/video2dataset/configs/panda70m.yaml) file to turn off the audio and increase download speed.

## Acknowledgements
The code for data downloading is built upon [video2dataset](https://github.com/iejMac/video2dataset).
Thanks for sharing the great codebase!
