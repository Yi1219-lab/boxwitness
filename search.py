"""Find the evidence clip for an order.

    python search.py ORD-1001          # list clips for one order
    python search.py ORD-1001 --play   # open the newest clip
    python search.py --all             # list everything
"""

import argparse
import datetime as dt
import os
import platform
import subprocess

from boxwitness.config import Config
from boxwitness.recorder import EvidenceIndex


def fmt(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)  # noqa: S606
    elif platform.system() == "Darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("order_id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    idx = EvidenceIndex(Config().db_path)
    rows = idx.all() if args.all or not args.order_id else idx.find(args.order_id)
    if not rows:
        print("No clips found.")
        return
    for oid, s, e, path, status, summary in rows:
        print(f"{oid:10s} {fmt(s)} ({e - s:5.1f}s)  {status:8s} {summary}\n           {path}")
    if args.play and args.order_id:
        open_file(rows[-1][3])


if __name__ == "__main__":
    main()
