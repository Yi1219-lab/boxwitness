"""Order list loading, the packing state machine, and verification."""

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


def load_orders(path):
    """orders.json format: {"ORD-1001": {"bottle": 1, "cup": 1}, ...}"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {oid: Counter({k: int(v) for k, v in items.items()})
            for oid, items in raw.items()}


@dataclass
class VerifyResult:
    status: str            # "PASS" | "MISSING" | "EXTRA" | "MISMATCH"
    missing: dict
    extra: dict

    def summary(self):
        if self.status == "PASS":
            return "PASS - all items packed"
        parts = []
        if self.missing:
            parts.append("missing " + ", ".join(f"{n}x {k}" for k, n in self.missing.items()))
        if self.extra:
            parts.append("extra " + ", ".join(f"{n}x {k}" for k, n in self.extra.items()))
        return f"{self.status} - " + "; ".join(parts)


def verify(expected, counted):
    expected, counted = Counter(expected), Counter(counted)
    missing = dict(expected - counted)
    extra = dict(counted - expected)
    if not missing and not extra:
        status = "PASS"
    elif missing and extra:
        status = "MISMATCH"
    elif missing:
        status = "MISSING"
    else:
        status = "EXTRA"
    return VerifyResult(status, missing, extra)


class State(Enum):
    IDLE = "IDLE"
    PACKING = "PACKING"


@dataclass
class Session:
    order_id: str
    expected: Counter
    started_at: float = field(default_factory=time.time)


class PackingStateMachine:
    """IDLE --scan(order)--> PACKING --scan(same order)--> verify --> IDLE

    Scanning a *different* order while PACKING closes the current one first,
    so a packer who forgets to close an order never loses evidence.
    """

    def __init__(self, orders):
        self.orders = orders
        self.state = State.IDLE
        self.session = None

    def on_scan(self, code):
        """Returns a list of actions: ("start", order_id) / ("finish", order_id) / ("unknown", code)."""
        actions = []
        if code not in self.orders:
            return [("unknown", code)]
        if self.state is State.PACKING:
            current = self.session.order_id
            actions.append(("finish", current))
            self._close()
            if code == current:
                return actions
        self.session = Session(code, self.orders[code])
        self.state = State.PACKING
        actions.append(("start", code))
        return actions

    def force_finish(self):
        if self.state is State.PACKING:
            oid = self.session.order_id
            self._close()
            return [("finish", oid)]
        return []

    def _close(self):
        self.state = State.IDLE
        self.session = None
