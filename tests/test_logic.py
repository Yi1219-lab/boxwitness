import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from boxwitness.counter import BoxCounter, Detection
from boxwitness.orders import PackingStateMachine, State, verify

ROI = (100, 100, 200, 200)


def det(tid, x, y, cls="bottle"):
    return Detection(tid, cls, 0.9, (x - 5, y - 5, x + 5, y + 5))


def test_counts_once_after_entering_box():
    c = BoxCounter(ROI, min_inside_frames=3)
    c.update([det(1, 50, 50)])                 # outside
    for _ in range(10):                        # inside for many frames
        c.update([det(1, 150, 150)])
    assert c.counts == Counter({"bottle": 1})


def test_requires_min_inside_frames():
    c = BoxCounter(ROI, min_inside_frames=3)
    c.update([det(1, 50, 50)])
    c.update([det(1, 150, 150)])
    c.update([det(1, 150, 150)])
    c.update([det(1, 50, 50)])                 # swept across, left again
    assert c.counts == Counter()


def test_reassigned_id_inside_box_not_double_counted():
    c = BoxCounter(ROI, min_inside_frames=2)
    c.update([det(1, 50, 50)])
    for _ in range(3):
        c.update([det(1, 150, 150)])
    for _ in range(5):                         # tracker lost it, new id 2 inside
        c.update([det(2, 150, 150)])
    assert c.counts == Counter({"bottle": 1})


def test_verify_statuses():
    assert verify({"bottle": 1}, {"bottle": 1}).status == "PASS"
    r = verify({"bottle": 1, "cup": 1}, {"bottle": 1})
    assert r.status == "MISSING" and r.missing == {"cup": 1}
    assert verify({"bottle": 1}, {"bottle": 2}).status == "EXTRA"
    assert verify({"bottle": 1}, {"cup": 1}).status == "MISMATCH"


def test_state_machine_flow():
    fsm = PackingStateMachine({"A": Counter(bottle=1), "B": Counter(cup=1)})
    assert fsm.on_scan("A") == [("start", "A")]
    assert fsm.state is State.PACKING
    assert fsm.on_scan("B") == [("finish", "A"), ("start", "B")]
    assert fsm.on_scan("B") == [("finish", "B")]
    assert fsm.state is State.IDLE
    assert fsm.on_scan("ZZZ") == [("unknown", "ZZZ")]
