"""Crowd grouping on top of person detections.

Idea
----
A "crowd" is a set of people that are close to each other. Two people are
linked when the distance between their box centres is small *relative to their
own size* (so the rule works for near and far people alike). Crowds are the
connected components of that link graph (this is single-linkage clustering).
A person with no neighbour is "isolated" and is NOT counted as a crowd.

Example from the task: a room with two groups of people and one person sitting
alone -> 2 crowds, 1 isolated person.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CrowdResult:
    labels: np.ndarray                      # crowd id (1..K) per person, -1 = isolated
    groups: list = field(default_factory=list)  # one dict per crowd
    n_persons: int = 0

    @property
    def n_crowds(self) -> int:
        return len(self.groups)

    @property
    def n_isolated(self) -> int:
        return int((self.labels == -1).sum())


def _find(parent: list, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def group_people(
    boxes,
    link_factor: float = 1.5,
    min_group_size: int = 2,
    max_scale_ratio: float = 2.5,
) -> CrowdResult:
    """Cluster person boxes (xyxy) into crowds.

    link_factor     : two people are linked if centre distance <= link_factor * mean person size
                      (size = sqrt(w*h)). Bigger value -> looser groups.
    min_group_size  : smallest number of people that counts as a crowd.
    max_scale_ratio : people whose sizes differ by more than this are at very different
                      depths, so they are not linked (avoids joining near + far people).
    """
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(b)
    labels = -np.ones(n, dtype=int)
    if n == 0:
        return CrowdResult(labels, [], 0)

    w = np.clip(b[:, 2] - b[:, 0], 1e-6, None)
    h = np.clip(b[:, 3] - b[:, 1], 1e-6, None)
    centres = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    size = np.sqrt(w * h)

    dist = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    mean_size = 0.5 * (size[:, None] + size[None, :])
    ratio = np.maximum(size[:, None], size[None, :]) / np.minimum(size[:, None], size[None, :])
    adjacent = (dist <= link_factor * mean_size) & (ratio <= max_scale_ratio)

    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adjacent[i, j]:
                ri, rj = _find(parent, i), _find(parent, j)
                if ri != rj:
                    parent[rj] = ri

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(_find(parent, i), []).append(i)

    big = [m for m in comps.values() if len(m) >= min_group_size]
    big.sort(key=lambda m: (-len(m), float(centres[m, 0].mean())))  # biggest first

    groups = []
    for gid, members in enumerate(big, start=1):
        labels[members] = gid
        gb = b[members]
        groups.append(
            {
                "id": gid,
                "members": members,
                "size": len(members),
                "bbox": (float(gb[:, 0].min()), float(gb[:, 1].min()),
                         float(gb[:, 2].max()), float(gb[:, 3].max())),
            }
        )
    return CrowdResult(labels, groups, n)
