# 🐼 Panda-70M: Dataset Dataloading
The section includes the csv files listing the data samples in Panda-70M and the code to download the videos.

**[Note] Please use the video2dataset tool from this repository to download the dataset. As the video2dataset from [the official repository](https://github.com/iejMac/video2dataset) cannot work with our csv format for now. We are working on making Panda-70M downloadable through the official video2dataset.**

## Data Splitting and Download Link
  | Split           | Download | # Source Videos | # Samples | Video Duration | Storage Space|
  |-----------------|----------|-----------------|-----------|----------------|--------------|
  | Training (full) | [link](https://drive.google.com/file/d/1DeODUcdJCEfnTjJywM-ObmrlVg-wsvwz/view?usp=sharing) (2.01 GB) | 3,779,763 | 70,723,513 | 167 khrs  | ~36 TB  |
  | Training (10M)  | [link](https://drive.google.com/file/d/1Lrsb65HTJ2hS7Iuy6iPCmjoc3abbEcAX/view?usp=sharing) (381 MB)  | 3,755,240 | 10,473,922 | 37.0 khrs | ~8.0 TB |
  | Training (2M)   | [link](https://drive.google.com/file/d/1jWTNGjb-hkKiPHXIbEA5CnFwjhA-Fq_Q/view?usp=sharing) (86.5 MB) | 800,000   | 2,400,000  | 7.56 khrs | ~1.6 TB |
  | Validation      | [link](https://drive.google.com/file/d/1cTCaC7oJ9ZMPSax6I4ZHvUT-lqxOktrX/view?usp=sharing) (803 KB)  | 2,000     | 6,000      | 18.5 hrs  | ~4.0 GB |
  | Testing         | [link](https://drive.google.com/file/d/1ee227tHEO-DT8AkX7y2q6-bfAtUL-yMI/view?usp=sharing) (803 KB)  | 2,000     | 6,000      | 18.5 hrs  | ~4.0 GB |
- Validation and testing set are collected from 2,000 source videos which do not appear in any training set to avoid testing information leakage. For each source video, we randomly sample 3 clips.
- Training set (10M) is the high-quality subset of training set (full). In the subset, we only sample at most 3 clips from a source video to increase diversity and the video-caption matching scores are all larger than 0.43 to guarantee a better caption quality.
- Training set (2M) is randomly sampled from training set (10M) and include 3 clips for each source video.
- **[Note 1]** The training csv files are too large and are compressed into zip files. Please `unzip` to get the csv files.
- **[Note 2]** We will remove the video samples from our dataset as long as you need it. Please contact tsaishienchen at gmail dot com for the request.
 
## Download Dataset
### Setup Repository and Enviroment
```
git clone https://github.com/tsaishien-chen/Panda-70M.git
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
              --save_additional_columns="[matching_score]" \
              --config="video2dataset/video2dataset/configs/panda_70M.yaml"
```
- **[Note 1]** If you get `HTTP Error 403: Forbidden` error, it might because your IP got banned. Please refer [this issue](https://github.com/yt-dlp/yt-dlp/issues/8785) and try to download the data by another IP.
- **[Note 2]** You will get `"status": "failed_to_download"` and `"error_message": "[Errno 2] No such file or directory: '/tmp/*.mp4'"`, if the YouTube video has been set to private or removed.

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
- Meta information includes matching score (confidence score of each video-caption pair), caption, video title / description / categories / subtitles, to name but a few.
- **[Note 1]** The dataset is unshuffled and the clips from a same long video would be stored into a shard. Please manually shuffle them if needed.
- **[Note 2]** The videos are resized into 360 px height. You can change `download_size` in the [config](./video2dataset/video2dataset/configs/panda_70M.yaml) file to get different video resolutions.

## Acknowledgements
The code for data downloading is built upon [video2dataset](https://github.com/iejMac/video2dataset).
Thanks for sharing the great codebase!
