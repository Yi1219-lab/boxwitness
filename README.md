# BoxWitness

**Edge-AI packing verification that turns every shipment into searchable evidence.**

ASUS UGen AI League 2026 · Battlefield Lightning (ASUS UGen300 / Hailo-10H) · Workplace AI

Author: Heng-Yi Li (李衡毅), NTU CSIE

---

## The problem

Online sellers lose money on "item missing" disputes. Many already film their packing desk, but that footage is not linked to orders, nobody checks the items before the box is sealed, and finding the right 40 seconds in a day of video takes far too long.

## What BoxWitness does

1. **Scan**: the packer shows the order's QR code. A clip starts recording, including 2 s of pre-roll.
2. **Count**: a YOLOv8n detector plus ByteTrack counts each item as it moves into the box region.
3. **Verify**: counted items are compared with the order. `PASS`, `MISSING`, `EXTRA` or `MISMATCH` is shown on screen.
4. **Index**: one clip per order, with the verdict burned into the video, stored in SQLite and searchable by order ID.

## Architecture

```
USB webcam
   │
   ├─► [CPU] QR order scan (OpenCV) ─► state machine (IDLE / PACKING)
   │
   ├─► [NPU*] YOLOv8n detection, every frame
   │        └─► [CPU] ByteTrack + box-ROI counter ─► order verifier
   │
   └─► [CPU] ring-buffer recorder ─► per-order clip ─► SQLite evidence index

* Stage I runs the detector on the laptop CPU (Ultralytics).
  Stage II moves it to the ASUS UGen300 (Hailo-10H) through boxwitness/detector.py.
```

**Cascade design.** The detector is small and runs every frame. Heavier work (the Stage II SKU-embedding model) runs only when an event happens, such as an item entering the box. This keeps the always-on cost low enough for a 2.5 W USB accelerator.

### Counting rule (`boxwitness/counter.py`)

A detector sees the same object in many frames, so we count **tracks**, not detections. A track is counted once, when:

1. it has been seen **outside** the box at least once, and
2. its centre then stays **inside** the box for `min_inside_frames` consecutive frames (default 4).

Rule 1 prevents double counting when the tracker loses an item already in the box and gives it a new ID. Rule 2 ignores items that are only swept across the box edge.

### Verification (`boxwitness/orders.py`)

Orders and counts are multisets: `missing = expected - counted`, `extra = counted - expected`.

## Project layout

```
app.py                 main loop (camera → detect/track → count → verify → record)
search.py              find a clip by order ID
orders.json            demo orders (COCO class names)
boxwitness/config.py    all tunable parameters
boxwitness/counter.py   tracking-based counting
boxwitness/orders.py    order loading, state machine, verification
boxwitness/recorder.py  pre-roll ring buffer, clip writer, SQLite index
boxwitness/detector.py  inference backends (Ultralytics now, Hailo planned)
boxwitness/ui.py        overlay drawn on screen and burned into clips
tools/make_qr.py       generate a QR image per order
tests/test_logic.py    unit tests for counting, verification and state machine
```

## Quick start (Windows, Python 3.10–3.12)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tools/make_qr.py          # writes qr/ORD-1001.png ...
python app.py                    # webcam 0; first run downloads yolov8n.pt
```

macOS / Linux: use `source .venv/bin/activate` instead.

**Using it**

- Open `qr/ORD-1001.png` on your phone and hold it up to the camera: the order starts.
- Move each item from the left side of the frame into the green **BOX** region on the right and hold it there for half a second.
- Show the same QR again to finish. The verdict appears at the bottom of the window.
- Keys: `n` start the next order without a QR, `f` finish the current order, `q` quit.

```powershell
python search.py ORD-1001          # list clips for that order
python search.py ORD-1001 --play   # open the newest clip
python search.py --all
```

Other sources: `python app.py --source 1` (second camera) or `python app.py --source demo.mp4`.

**Detector classes.** The Stage I model is COCO-pretrained, so demo orders use COCO classes that are easy to find at home: `bottle`, `cup`, `cell phone`, `scissors`, `book`, `remote`. Edit `orders.json` to match what you have.

## Tests and metrics

```powershell
pip install pytest
python -m pytest -q tests
```

On exit, `app.py` writes `metrics.csv` with mean, p50 and p95 inference time and end-to-end FPS.

Measured on a 2-vCPU x86 machine, CPU only, YOLOv8n at 640 px: about 80 ms median detection + tracking time per frame and about 10 FPS end to end. This compute cost is what the Hailo-10H removes in Stage II.

## Stage II plan (UGen300)

| Week | Work |
|---|---|
| Nov 6–12 | HailoRT setup, YOLOv8n HEF from the Hailo Model Zoo, measure FPS and latency over USB |
| Nov 13–19 | Implement `HailoDetector` behind the existing backend interface; build a 20-product test set |
| Nov 20–26 | Event-triggered SKU embedding on the NPU; occlusion handling; packer UI |
| Nov 27–Dec 4 | Field test at a real seller's packing desk; report accuracy, power, retrieval time |

## Related products

Packing-video evidence is an existing product category:

- **Recording-only apps** (for example Shopify apps that save one packing video per order) link video to orders but do not check what went into the box.
- **Enterprise pack-station systems** (for example Rabot's packing video verification) integrate with a warehouse management system and advertise vision AI that flags items that do not match the order.

BoxWitness aims at the gap between them: item verification for small sellers who have no WMS, running on a single low-power USB accelerator plugged into an existing laptop, with all footage kept on-site.

## Known limitations

- The COCO detector recognises generic categories ("bottle"), not specific products. Few-shot SKU matching is planned for Stage II.
- Hands and box flaps can hide items. Camera placement and the N-frame rule reduce this.
- Stage I has not run on the UGen300 yet; the Hailo backend is a documented stub.

## License

MIT
