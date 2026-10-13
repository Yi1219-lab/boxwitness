"""BoxWitness main loop.

    python app.py                    # webcam 0
    python app.py --source 1         # another webcam
    python app.py --source demo.mp4  # a recorded video
    python app.py --headless         # no window (for testing)

Keys (in the video window):
    n  start the next order without scanning (fallback if QR reading is flaky)
    f  finish the current order
    q  quit
"""

import argparse
import csv
import platform
import statistics
import time

import cv2

from boxwitness.config import Config
from boxwitness.counter import BoxCounter
from boxwitness.detector import UltralyticsDetector
from boxwitness.orders import PackingStateMachine, State, load_orders, verify
from boxwitness.recorder import ClipRecorder, EvidenceIndex
from boxwitness.ui import draw, draw_banner


def open_source(src, w, h):
    """Open a webcam index or a video file.

    On Windows, webcams are reachable through two APIs: Media Foundation (MSMF)
    and DirectShow (DSHOW). Which one works depends on the driver, so try both
    and fall back to OpenCV's automatic choice. A backend only counts as working
    if it actually returns a frame, because some report isOpened() and then
    deliver nothing.
    """
    if not src.isdigit():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video file: {src}")
        return cap

    if platform.system() == "Windows":
        backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
    else:
        backends = [("ANY", cv2.CAP_ANY)]

    for name, backend in backends:
        cap = cv2.VideoCapture(int(src), backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok = False
        for _ in range(10):            # some cameras need a few frames to warm up
            ok, _frame = cap.read()
            if ok:
                break
        if ok:
            print(f"[CAMERA] index {src} opened with {name}")
            return cap
        cap.release()

    raise SystemExit(
        f"Cannot open webcam {src}. Check that no other app (Teams, Zoom, LINE, a browser tab) "
        "is using the camera, and that Windows Settings > Privacy & security > Camera allows "
        "desktop apps. Or try another index: python app.py --source 1")


def roi_to_px(roi, W, H):
    x1, y1, x2, y2 = roi
    return (x1 * W, y1 * H, x2 * W, y2 * H)


def write_metrics(path, infer_ms, loop_ms):
    if not infer_ms:
        return None
    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]
    summary = {
        "frames": len(infer_ms),
        "infer_ms_mean": statistics.mean(infer_ms),
        "infer_ms_p50": pct(infer_ms, 50),
        "infer_ms_p95": pct(infer_ms, 95),
        "loop_ms_mean": statistics.mean(loop_ms),
        "fps_mean": 1000.0 / statistics.mean(loop_ms),
        "host": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(summary.keys())
        w.writerow(summary.values())
    return summary


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=cfg.source)
    ap.add_argument("--model", default=cfg.model_path)
    ap.add_argument("--orders", default=cfg.orders_path)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    orders = load_orders(args.orders)
    order_queue = list(orders.keys())
    fsm = PackingStateMachine(orders)
    det = UltralyticsDetector(args.model, cfg.imgsz, cfg.conf, cfg.tracker, cfg.ignore_classes)
    qr = cv2.QRCodeDetector()
    index = EvidenceIndex(cfg.db_path)

    cap = open_source(str(args.source), cfg.frame_width, cfg.frame_height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    H, W = frame.shape[:2]
    roi_px = roi_to_px(cfg.box_roi, W, H)
    counter = BoxCounter(roi_px, cfg.min_inside_frames)
    recorder = ClipRecorder(cfg.clips_dir, cfg.record_fps, cfg.preroll_s)

    last_code, last_code_t = None, 0.0
    banner, banner_until = None, 0.0
    fps_ema, infer_ms_list, loop_ms_list = 0.0, [], []
    frame_idx = 0

    def finish(order_id, session_started, expected):
        nonlocal banner, banner_until
        res = verify(expected, counter.counts)
        verdict = None
        if recorder.last_frame is not None:
            verdict = draw_banner(recorder.last_frame.copy(),
                                  f"{order_id}: {res.summary()}", res.status == "PASS")
        path = recorder.stop(verdict)
        index.add(order_id, session_started, time.time(), path, expected, counter.counts, res)
        banner = (f"{order_id}: {res.summary()}", res.status == "PASS")
        banner_until = time.time() + 3.0
        print(f"[FINISH] {order_id} -> {res.summary()}  clip={path}")

    def apply(actions, session_before):
        for kind, oid in actions:
            if kind == "finish":
                finish(oid, session_before.started_at, session_before.expected)
            elif kind == "start":
                counter.reset()
                path = recorder.start(oid, (W, H))
                print(f"[START ] {oid} recording -> {path}")
                if oid in order_queue:
                    order_queue.remove(oid)
            elif kind == "unknown":
                print(f"[WARN  ] unknown code: {oid}")

    while ok:
        t_loop = time.perf_counter()

        # 1) order scanning (cheap, every N frames)
        if frame_idx % cfg.qr_every_n_frames == 0:
            code, _, _ = qr.detectAndDecode(frame)
            now = time.time()
            if code and not (code == last_code and now - last_code_t < cfg.qr_cooldown_s):
                last_code, last_code_t = code, now
                prev = fsm.session
                apply(fsm.on_scan(code), prev)

        # 2) always-on detection + tracking
        dets, infer_ms = det.track(frame)
        infer_ms_list.append(infer_ms)

        # 3) count items entering the box (only while packing)
        if fsm.state is State.PACKING:
            for item in counter.update(dets, frame_idx):
                print(f"[COUNT ] +1 {item}  totals={dict(counter.counts)}")

        # 4) overlay + evidence recording
        show_banner = banner if time.time() < banner_until else None
        vis = draw(frame, roi_px, dets, fsm.state.value, fsm.session,
                   counter.counts, show_banner, fps_ema, infer_ms)
        recorder.push(vis)

        if not args.headless:
            cv2.imshow("BoxWitness", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                prev = fsm.session
                apply(fsm.force_finish(), prev)
            if key == ord("n") and fsm.state is State.IDLE and order_queue:
                apply(fsm.on_scan(order_queue[0]), None)

        loop_ms = (time.perf_counter() - t_loop) * 1000.0
        loop_ms_list.append(loop_ms)
        fps_ema = 0.9 * fps_ema + 0.1 * (1000.0 / max(loop_ms, 1e-3))
        frame_idx += 1
        ok, frame = cap.read()

    prev = fsm.session
    apply(fsm.force_finish(), prev)
    cap.release()
    cv2.destroyAllWindows()
    summary = write_metrics(cfg.metrics_path, infer_ms_list, loop_ms_list)
    if summary:
        print("[METRIC] " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in summary.items()))


if __name__ == "__main__":
    main()
