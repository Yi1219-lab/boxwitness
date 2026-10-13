"""Evidence recording: pre-roll ring buffer, per-order clips, SQLite index.

Why a ring buffer
-----------------
The system only learns an order has started when the packer scans its code,
but the moments just before the scan (picking the item up) are part of the
evidence. A fixed-size deque keeps the last `preroll_s` seconds of frames
at all times, so each clip can begin a little *before* the scan.
"""

import json
import os
import sqlite3
import time
from collections import deque

import cv2

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL NOT NULL,
    clip_path   TEXT NOT NULL,
    expected    TEXT NOT NULL,
    counted     TEXT NOT NULL,
    status      TEXT NOT NULL,
    summary     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_order ON clips(order_id);
"""


class EvidenceIndex:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def add(self, order_id, started_at, ended_at, clip_path, expected, counted, result):
        self.conn.execute(
            "INSERT INTO clips (order_id, started_at, ended_at, clip_path, expected, counted, status, summary)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (order_id, started_at, ended_at, clip_path,
             json.dumps(dict(expected)), json.dumps(dict(counted)),
             result.status, result.summary()))
        self.conn.commit()

    def find(self, order_id):
        cur = self.conn.execute(
            "SELECT order_id, started_at, ended_at, clip_path, status, summary"
            " FROM clips WHERE order_id = ? ORDER BY started_at", (order_id,))
        return cur.fetchall()

    def all(self):
        cur = self.conn.execute(
            "SELECT order_id, started_at, ended_at, clip_path, status, summary"
            " FROM clips ORDER BY started_at")
        return cur.fetchall()


class ClipRecorder:
    def __init__(self, clips_dir, fps, preroll_s):
        os.makedirs(clips_dir, exist_ok=True)
        self.clips_dir = clips_dir
        self.fps = fps
        self.buffer = deque(maxlen=max(1, int(fps * preroll_s)))
        self.writer = None
        self.path = None
        self.last_frame = None

    @property
    def recording(self):
        return self.writer is not None

    def push(self, frame):
        """Call once per frame with the annotated frame."""
        if self.writer is not None:
            self.writer.write(frame)
            self.last_frame = frame
        else:
            self.buffer.append(frame.copy())

    def start(self, order_id, frame_size):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in order_id)
        self.path = os.path.join(self.clips_dir, f"{safe}_{stamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, frame_size)
        for f in self.buffer:          # flush pre-roll into the clip
            self.writer.write(f)
        self.buffer.clear()
        return self.path

    def stop(self, verdict_frame=None, hold_s=2.0):
        """Close the clip. If `verdict_frame` is given, it is held at the end of
        the clip so the evidence video ends with the verification result."""
        path = self.path
        if self.writer is not None:
            if verdict_frame is not None:
                for _ in range(int(self.fps * hold_s)):
                    self.writer.write(verdict_frame)
            self.writer.release()
        self.writer, self.path, self.last_frame = None, None, None
        return path
