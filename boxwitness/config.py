"""Runtime configuration. Edit these values instead of hunting through the code."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Input ---------------------------------------------------------
    source: str = "0"                 # webcam index ("0") or a video file path
    frame_width: int = 1280
    frame_height: int = 720

    # --- Detection (stage 1, always-on) --------------------------------
    model_path: str = "yolov8n.pt"    # COCO-pretrained, auto-downloaded on first run
    imgsz: int = 640
    conf: float = 0.35
    tracker: str = "bytetrack.yaml"   # Ultralytics built-in ByteTrack config
    ignore_classes: tuple = ("person", "dining table", "chair", "couch", "bed")

    # --- Box ROI, as fractions of the frame (x1, y1, x2, y2) ------------
    # The "box" is the region where an item counts as packed.
    box_roi: tuple = (0.55, 0.30, 0.95, 0.95)

    # An item is counted when its track was seen OUTSIDE the box at least once
    # and then stays INSIDE for this many consecutive frames.
    min_inside_frames: int = 4

    # --- Order scanning ------------------------------------------------
    qr_every_n_frames: int = 3        # QR decode is cheap but not free
    qr_cooldown_s: float = 2.5        # ignore repeated reads of the same code

    # --- Evidence recording --------------------------------------------
    clips_dir: str = "clips"
    db_path: str = "boxwitness.db"
    preroll_s: float = 2.0            # seconds kept before the order starts
    record_fps: float = 15.0

    # --- Metrics -------------------------------------------------------
    metrics_path: str = "metrics.csv"

    orders_path: str = "orders.json"
    extra: dict = field(default_factory=dict)
