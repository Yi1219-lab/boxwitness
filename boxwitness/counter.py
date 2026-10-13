"""Counts items placed into the box ROI using tracker IDs.

Why tracking is required
------------------------
A detector runs on every frame independently. One bottle that stays in view
for 2 s at 30 fps is "detected" ~60 times. Counting detections would grossly
over-count. ByteTrack links detections across frames into a persistent
track ID, so we can count *events* ("track 7 entered the box") instead of
*frames*.

Counting rule
-------------
A track is counted once, when
  1. it has been observed OUTSIDE the box at least once, and
  2. its box-centre then stays INSIDE the box for `min_inside_frames`
     consecutive frames.

Rule 1 prevents double counting when the tracker loses an item that is
already lying in the box and re-assigns it a new ID (the new ID was never
outside). Rule 2 filters out items that are only swept across the box edge.
"""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class _TrackState:
    cls_name: str
    seen_outside: bool = False
    inside_streak: int = 0
    counted: bool = False


@dataclass
class Detection:
    track_id: int
    cls_name: str
    conf: float
    xyxy: tuple  # pixel coordinates (x1, y1, x2, y2)

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class BoxCounter:
    roi_px: tuple
    min_inside_frames: int = 4
    _tracks: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    events: list = field(default_factory=list)  # (frame_idx, track_id, cls)

    def reset(self):
        self._tracks.clear()
        self.counts = Counter()
        self.events = []

    def update(self, detections, frame_idx=0):
        """Feed one frame of tracked detections. Returns newly counted items."""
        newly = []
        for d in detections:
            st = self._tracks.get(d.track_id)
            if st is None:
                st = _TrackState(cls_name=d.cls_name)
                self._tracks[d.track_id] = st
            inside = point_in_rect(d.center, self.roi_px)
            if not inside:
                st.seen_outside = True
                st.inside_streak = 0
                continue
            st.inside_streak += 1
            if (not st.counted and st.seen_outside
                    and st.inside_streak >= self.min_inside_frames):
                st.counted = True
                self.counts[st.cls_name] += 1
                self.events.append((frame_idx, d.track_id, st.cls_name))
                newly.append(st.cls_name)
        return newly
