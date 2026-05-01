"""FFPose: pure-PyTorch pose inference and training, no mm-* dependencies."""
from .codec import SimCCDecoder, get_simcc_maximum
from .detector import Detections, PersonDetector
from .hrformer_pose import (
    HRFORMER_POSE_COCO_256x192,
    HRFormerPose,
    HRFormerPoseConfig,
    HRFormerPoseInferencer,
)
from .hrnet_pose import (
    HRNET_POSE_COCO_256x192,
    HRNetPose,
    HRNetPoseConfig,
    HRNetPoseInferencer,
)
from .inference import PoseResult, RTMPoseInferencer
from .litehrnet_pose import (
    LITEHRNET_POSE_COCO_256x192,
    LiteHRNetPose,
    LiteHRNetPoseConfig,
    LiteHRNetPoseInferencer,
)
from .model import RTMPOSE_COCO_256x192, RTMPose, RTMPoseConfig
from .pipeline import FullFrameResult, TopDownPosePipeline
from .swin_pose import (
    SWIN_POSE_COCO_256x192,
    SwinPose,
    SwinPoseConfig,
    SwinPoseInferencer,
)
from .visualization import draw_skeleton, draw_skeletons
from .vitpose import VITPOSE_COCO_256x192, ViTPose, ViTPoseConfig, ViTPoseInferencer

__all__ = [
    # RTMPose
    "RTMPose", "RTMPoseConfig", "RTMPOSE_COCO_256x192", "RTMPoseInferencer",
    # ViTPose
    "ViTPose", "ViTPoseConfig", "VITPOSE_COCO_256x192", "ViTPoseInferencer",
    # HRNet
    "HRNetPose", "HRNetPoseConfig", "HRNET_POSE_COCO_256x192", "HRNetPoseInferencer",
    # Swin
    "SwinPose", "SwinPoseConfig", "SWIN_POSE_COCO_256x192", "SwinPoseInferencer",
    # HRFormer
    "HRFormerPose", "HRFormerPoseConfig", "HRFORMER_POSE_COCO_256x192", "HRFormerPoseInferencer",
    # LiteHRNet
    "LiteHRNetPose", "LiteHRNetPoseConfig", "LITEHRNET_POSE_COCO_256x192", "LiteHRNetPoseInferencer",
    # Common
    "PoseResult", "SimCCDecoder", "get_simcc_maximum",
    "PersonDetector", "Detections",
    "TopDownPosePipeline", "FullFrameResult",
    "draw_skeleton", "draw_skeletons",
]
