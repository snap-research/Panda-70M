"""
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""

from .audio_rate_subsampler import AudioRateSubsampler
from .clipping_subsampler import ClippingSubsampler, _get_seconds, _split_time_frame
from .frame_subsampler import FrameSubsampler
from .ffprobe_subsampler import FFProbeSubsampler
from .noop_subsampler import NoOpSubsampler
from .resolution_subsampler import ResolutionSubsampler
from .cut_detection_subsampler import CutDetectionSubsampler
from .optical_flow_subsampler import OpticalFlowSubsampler
from .whisper_subsampler import WhisperSubsampler
from .caption_subsampler import CaptionSubsampler
