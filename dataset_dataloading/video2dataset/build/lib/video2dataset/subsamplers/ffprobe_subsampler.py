"""extracts basic video compression metadata."""
import os
import json
import subprocess
import tempfile

from .subsampler import Subsampler

# TODO: figuer out why this is so slow (12 samples/s)
class FFProbeSubsampler(Subsampler):
    """
    Extracts metadata from bytes.
    Args:
        extract_keyframes (bool): Whether to extract keyframe timestamps.
    """

    def __init__(self, extract_keyframes=False):
        self.extract_keyframes = extract_keyframes

    def __call__(self, streams, metadata):
        # TODO: this should also work for audio (maybe others)
        video_bytes = streams["video"][0]
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                f.write(video_bytes)
            try:
                command = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    f"{tmpdir}/input.mp4",
                ]

                if self.extract_keyframes:
                    command.extend(["-select_streams", "v:0", "-show_entries", "packet=pts_time,flags"])

                process = subprocess.run(command, capture_output=True, text=True, check=True)
                video_metadata = json.loads(process.stdout)

                if self.extract_keyframes:
                    keyframe_info = [entry for entry in video_metadata["packets"] if "K" in entry.get("flags", "")]
                    keyframe_timestamps = [float(entry["pts_time"]) for entry in keyframe_info]
                    if "duration" in video_metadata["format"]:
                        duration = float(video_metadata["format"]["duration"])
                        keyframe_timestamps.append(duration)
                    video_metadata["keyframe_timestamps"] = keyframe_timestamps
                    video_metadata.pop("packets")  # Don't need it anymore
                metadata["video_metadata"] = video_metadata

            except Exception as err:  # pylint: disable=broad-except
                return streams, metadata, str(err)

        return streams, metadata, None
