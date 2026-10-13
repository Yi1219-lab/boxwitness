"""On-screen overlay. The overlay is burned into the recorded clips on purpose:
the evidence video then shows exactly what the system counted and when."""

import time

import cv2

GREEN, RED, AMBER, WHITE, GRAY = (60, 200, 60), (50, 50, 230), (0, 170, 255), (255, 255, 255), (90, 90, 90)


def _panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = cv2.addWeighted(sub, 1 - alpha, sub * 0, alpha, 0)


def draw(frame, roi_px, dets, state, session, counts, banner=None, fps=0.0, infer_ms=0.0):
    x1, y1, x2, y2 = map(int, roi_px)
    roi_color = GREEN if state == "PACKING" else GRAY
    cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
    cv2.putText(frame, "BOX", (x1 + 8, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

    for d in dets:
        a, b, c, e = map(int, d.xyxy)
        cv2.rectangle(frame, (a, b), (c, e), AMBER, 2)
        cv2.putText(frame, f"{d.cls_name} #{d.track_id}", (a, max(18, b - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, AMBER, 2)

    # status panel
    lines = [f"State: {state}"]
    if session is not None:
        lines.append(f"Order: {session.order_id}")
        for item, need in sorted(session.expected.items()):
            got = counts.get(item, 0)
            mark = "OK" if got >= need else "  "
            lines.append(f" [{mark}] {item}: {got}/{need}")
        extra = {k: v for k, v in counts.items() if k not in session.expected}
        for item, got in extra.items():
            lines.append(f" [!!] {item}: {got}/0")
    else:
        lines.append("Scan an order QR to start")
    lines.append(f"{fps:4.1f} FPS | infer {infer_ms:5.1f} ms")
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))

    h = 28 * len(lines) + 16
    _panel(frame, 10, 10, 360, h)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (22, 40 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)

    if banner is not None:
        draw_banner(frame, *banner)
    return frame


def draw_banner(frame, text, ok):
    color = GREEN if ok else RED
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, H - 70), (W, H), color, -1)
    cv2.putText(frame, text, (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3)
    return frame
