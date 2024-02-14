"""creates a subset of an existing dataset inside the sample dimension"""
import time
import json
import pyarrow as pa
import traceback

import fsspec
import numpy as np
import webdataset as wds
from typing import List, Any

from video2dataset.dataloader import get_video_dataset
from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    FFProbeSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
)


class SubsetWorker:
    """The loader class reads the shards, then the selected data is chosen and writen by the writer"""

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        encode_formats,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = True
        self.output_folder = output_folder
        self.encode_formats = encode_formats
        self.config = config

        self.clipping_subsampler = ClippingSubsampler(
            5,  # oom_clip_count
            encode_formats,
            **self.config["subsampling"].get("ClippingSubsampler", {"args": {}})["args"],
        )
        need_keyframes = self.clipping_subsampler.precision == "keyframe_adjusted"

        self.ffprobe_subsampler = None
        if "FFProbeSubsampler" in self.config["subsampling"] or need_keyframes:
            self.ffprobe_subsampler = FFProbeSubsampler(
                **self.config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"]
            )
            self.ffprobe_subsampler.extract_keyframes |= need_keyframes

        self.cut_detector = None
        self.cuts_are_clips = False
        if "CutDetectionSubsampler" in self.config["subsampling"]:
            if "args" in self.config["subsampling"]["CutDetectionSubsampler"]:
                self.cut_detector = CutDetectionSubsampler(
                    **self.config["subsampling"]["CutDetectionSubsampler"]["args"]
                )
            self.cuts_are_clips = self.config["subsampling"]["CutDetectionSubsampler"].get("cuts_are_clips", False)

        self.noop_subsampler = NoOpSubsampler()

        video_subsamplers: List[Any] = []
        if "ResolutionSubsampler" in self.config["subsampling"]:
            video_subsamplers.append(ResolutionSubsampler(**self.config["subsampling"]["ResolutionSubsampler"]["args"]))
        if "FrameSubsampler" in self.config["subsampling"]:
            video_subsamplers.append(FrameSubsampler(**self.config["subsampling"]["FrameSubsampler"]["args"]))

        audio_subsamplers: List[Any] = []
        if "AudioRateSubsampler" in self.config["subsampling"]:
            audio_subsamplers.append(AudioRateSubsampler(**self.config["subsampling"]["AudioRateSubsampler"]["args"]))
        self.subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    def __call__(
        self,
        row,
    ):
        try:
            self.process_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def process_shard(
        self,
        row,
    ):
        """Function to start an video processing in one process"""

        shard, shard_id = row
        start_time = time.time()

        try:
            fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")

            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception as e:  # pylint: disable=broad-except,unused-variable
            fields = [
                pa.field("key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("error_message", pa.string()),
            ]
            schema = pa.schema(fields)

        status_dict = CappedCounter()

        # The subsamplers might change the output format, so we need to update the writer
        writer_encode_formats = self.encode_formats.copy()
        if self.subsamplers["audio"]:
            writer_encode_formats["audio"] = self.subsamplers["audio"][0].encode_formats["audio"]
        if self.subsamplers["video"]:
            writer_encode_formats["video"] = self.subsamplers["video"][0].encode_formats["video"]

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.config["storage"]["oom_shard_count"],
            schema,
            writer_encode_formats,
        )

        successes = 0
        failed = {
            "failed_to_download": 0,
            "failed_to_subsample": 0,
        }
        error_message = None

        dataloader = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs={},
            enforce_additional_keys=[],
            handler=wds.warn_and_continue,
        )
        count = 0
        for sample in dataloader:
            try:
                count += 1
                key = sample["__key__"]
                caption = sample.get("txt", b"").decode("utf-8")
                meta = json.loads(sample.get("json", b"{}").decode("utf-8"))
                streams = {}
                for mod, fmt in self.encode_formats.items():
                    streams[mod] = [sample[fmt]]

                if self.ffprobe_subsampler is not None:
                    streams, meta, error_message = self.ffprobe_subsampler(streams, meta)
                    if error_message is not None:
                        raise Exception("failed_to_subsample")

                if self.config["storage"]["captions_are_subtitles"]:  # create clips
                    subtitles = meta["yt_meta_dict"]["subtitles"]
                    meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                elif self.cut_detector is not None:  # apply cut detection to get clips
                    streams, cuts, error_message = self.cut_detector(streams)
                    if error_message is not None:
                        raise Exception("failed_to_subsample")

                    meta["cuts"] = cuts

                if self.cuts_are_clips:
                    cuts = meta["cuts"]
                    native_fps = cuts["original_fps"]
                    meta["clips"] = (np.array(cuts["cuts_original_fps"]) / native_fps).tolist()

                # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                broadcast_subsampler = (
                    self.clipping_subsampler
                    if (self.config["storage"]["captions_are_subtitles"] or self.cuts_are_clips)
                    else self.noop_subsampler
                )
                subsampled_streams, metas, error_message = broadcast_subsampler(streams, meta)
                if error_message is not None:
                    meta["clips"] = []
                    raise Exception("failed_to_subsample")

                for modality in list(subsampled_streams.keys()):
                    for modality_subsampler in self.subsamplers[modality]:
                        subsampled_streams, metas, error_message = modality_subsampler(subsampled_streams, metas)

                if error_message is not None:
                    raise Exception("failed_to_subsample")

                successes += 1
                status = "success"
                status_dict.increment(status)
                subsampled_streams_list = [dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())]
                if len(subsampled_streams_list) == 0:  # no audio or video, just write meta
                    meta["status"] = status
                    sample_writer.write(
                        {},
                        key,
                        caption,
                        meta,
                    )
                    continue

                for subsampled_streams, meta in zip(subsampled_streams_list, metas):
                    meta["status"] = status

                    text_caption = caption
                    if self.config["storage"]["captions_are_subtitles"]:
                        text_caption = meta.get("clip_subtitles")[0]["lines"][0]

                    sample_writer.write(
                        subsampled_streams,
                        meta["key"],
                        text_caption,
                        meta,
                    )

            except Exception as err:  # pylint: disable=broad-except
                status = str(err)
                if status.startswith("failed_to_"):
                    failed[status] += 1
                    status_dict.increment(error_message)
                    meta["status"] = status
                    meta["error_message"] = error_message
                    sample_writer.write(
                        {},
                        key,
                        caption,
                        meta,
                    )
                else:
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")

        sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            0,  # failed to download
            failed["failed_to_subsample"],
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.config["storage"]["oom_shard_count"],
        )
