"""Unit tests for the crowd grouping logic (no model needed).  Run:  python -m pytest tests -q
or simply:  python tests/test_crowd.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.crowd import group_people


def person(cx, cy, w=60, h=170):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_two_groups_one_isolated():
    # the exact scenario from the task: 2 groups + 1 person sitting alone
    left = [person(150 + 70 * i, 300) for i in range(4)]      # group A (4 people)
    right = [person(900 + 70 * i, 320) for i in range(3)]     # group B (3 people)
    alone = [person(520, 310)]                                # isolated
    r = group_people(left + right + alone)
    assert r.n_crowds == 2
    assert r.n_isolated == 1
    assert sorted(g["size"] for g in r.groups) == [3, 4]
    assert r.labels[-1] == -1


def test_no_people():
    r = group_people(np.zeros((0, 4)))
    assert r.n_crowds == 0 and r.n_persons == 0


def test_all_isolated():
    r = group_people([person(100, 200), person(700, 200), person(1300, 200)])
    assert r.n_crowds == 0 and r.n_isolated == 3


def test_single_big_crowd():
    r = group_people([person(100 + 65 * i, 300) for i in range(10)])
    assert r.n_crowds == 1 and r.groups[0]["size"] == 10


def test_near_and_far_not_merged():
    near = person(500, 500, w=300, h=800)     # large (close to camera)
    far = person(540, 480, w=30, h=85)        # tiny (far away) but overlapping position
    r = group_people([near, far])
    assert r.n_crowds == 0


def test_min_group_size():
    r = group_people([person(100, 200), person(160, 200)], min_group_size=3)
    assert r.n_crowds == 0 and r.n_isolated == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
