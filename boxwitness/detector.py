"""Inference backends.

Stage I (this prototype) runs YOLOv8n + ByteTrack on the laptop CPU via
Ultralytics. The `Detector` interface is deliberately small so that the
Stage II backend for the ASUS UGen300 (Hailo-10H) can be dropped in without
touching the application logic.
"""

import time

from .counter import Detection


class Detector:
    def track(self, frame):
        """Return (list[Detection], inference_ms)."""
        raise NotImplementedError


class UltralyticsDetector(Detector):
    def __init__(self, model_path, imgsz=640, conf=0.35, tracker="bytetrack.yaml",
                 ignore_classes=()):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.imgsz, self.conf, self.tracker = imgsz, conf, tracker
        self.keep = [i for i, n in self.names.items() if n not in set(ignore_classes)]

    def track(self, frame):
        t0 = time.perf_counter()
        res = self.model.track(frame, persist=True, tracker=self.tracker,
                               imgsz=self.imgsz, conf=self.conf,
                               classes=self.keep, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        dets = []
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            for xyxy, cls, conf, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(),
                                            boxes.conf.tolist(), boxes.id.tolist()):
                dets.append(Detection(int(tid), self.names[int(cls)], float(conf), tuple(xyxy)))
        return dets, ms


class HailoDetector(Detector):
    """Stage II backend for ASUS UGen300 (Hailo-10H). NOT IMPLEMENTED YET.

    Plan:
      1. Export YOLOv8n to ONNX, compile to a .hef with the Hailo Dataflow
         Compiler (or use the pre-compiled YOLOv8n from the Hailo Model Zoo).
      2. Run inference through HailoRT (Python API) on the UGen300.
      3. Keep ByteTrack on the host CPU: it only processes a few dozen boxes
         per frame, so it is cheap; the NPU handles the convolution-heavy part.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Hailo backend is planned for Stage II, when the UGen300 hardware is available.")
