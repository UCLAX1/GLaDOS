"""
depth.py — how far away is each tracked person (ZED SDK depth, metres).

Takes the median depth of a patch in the middle of each person box (upper
torso: rows 25-50% of the box, central 30% of its width), so background
pixels at the box edges don't pull the value. Only available with --source zed.

Result: {track_id: metres}
"""

import numpy as np

from workers import Worker


class DepthWorker(Worker):
    key, label, hotkey = "depth", "Depth distance", "5"
    max_hz = 10

    def load(self):
        if not self.source.has_depth:
            raise RuntimeError("needs the ZED SDK source (--source zed)")

    def process(self, frame):
        track = self.hub.get("track")
        people = track.fresh_result() if track else None
        if frame.depth is None or not people:
            return {}
        out = {}
        h, w = frame.depth.shape[:2]
        for p in people:
            x1, y1, x2, y2 = p["box"]
            bw, bh = x2 - x1, y2 - y1
            u1, u2 = int(x1 + 0.35 * bw), int(x2 - 0.35 * bw)
            v1, v2 = int(y1 + 0.25 * bh), int(y1 + 0.50 * bh)
            patch = frame.depth[max(v1, 0):min(v2, h), max(u1, 0):min(u2, w)]
            patch = patch[np.isfinite(patch)]
            if patch.size:
                out[p["id"]] = float(np.median(patch))
        return out
