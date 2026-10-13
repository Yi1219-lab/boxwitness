"""Generate one QR image per order in orders.json (show them on your phone or print them).

    python tools/make_qr.py
"""

import json
import os
import sys

import qrcode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(orders_path=os.path.join(ROOT, "orders.json"), out_dir=os.path.join(ROOT, "qr")):
    os.makedirs(out_dir, exist_ok=True)
    with open(orders_path, encoding="utf-8") as f:
        orders = json.load(f)
    for oid in orders:
        img = qrcode.make(oid, box_size=12, border=4)
        path = os.path.join(out_dir, f"{oid}.png")
        img.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
