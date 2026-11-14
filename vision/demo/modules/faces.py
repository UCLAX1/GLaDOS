"""
faces.py — who is it? Face recognition from vision/face_memory.py.

Uses that script's InsightFace model (buffalo_l) and known_faces/*.npy
format. Two changes for running alongside everything else:
  * known faces are loaded once (and after a registration), not every frame
  * it only runs when needed: a new tracked person appeared, or every PERIOD_S

register(name) collects NUM_SAMPLES varied samples of the largest face over
the next frames, saves known_faces/<name>.npy and starts recognising it.

Result: [{"name": "arjun" | "unknown", "score": 0.62, "box": (x1, y1, x2, y2)}]
"""

import os
import re
import time

import numpy as np

from workers import JETSON, Worker, onnx_threads

PERIOD_S = float(os.environ.get("DEMO_FACE_PERIOD", "1.0"))
SAMPLE_DIFF = 0.995       # skip samples this similar to one already taken (as face_memory)


class FaceWorker(Worker):
    key, label, hotkey = "face", "Face recognition", "3"

    def load(self):
        with onnx_threads(2 if JETSON else 0):
            import face_memory as fm             # builds buffalo_l on import
        # buffalo_l also runs 2 landmark models and age/gender on every face;
        # recognition only needs the detector + embedding (~2x faster without them)
        fm.app.models = {k: m for k, m in fm.app.models.items() if k in ("detection", "recognition")}
        self.fm = fm
        self.known = {}
        self.seen_ids = set()
        self.registering = None                  # name being registered
        self.samples = []
        self.reload_known()

    def reload_known(self):
        known = {}
        for f in sorted(os.listdir(self.fm.FACES_DIR)):
            if f.endswith(".npy"):
                e = np.load(os.path.join(self.fm.FACES_DIR, f), allow_pickle=False)
                known[os.path.splitext(f)[0]] = e.reshape(1, -1) if e.ndim == 1 else e
        self.known = known
        self.message = f"{len(known)} known: {', '.join(known) or 'none'}"

    def register(self, name: str) -> str:
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            return "name must be letters, numbers, _ or -"
        if self.state != "running":
            return f"face recognition is {self.state}"
        self.samples, self.registering = [], name
        return f"registering {name}: look at the camera and slowly turn your head"

    def want(self, frame):
        if self.registering:
            return True
        if time.time() - self.result_t >= PERIOD_S:
            return True
        track = self.hub.get("track")
        ids = {p["id"] for p in (track.fresh_result() or [])} if track else set()
        return bool(ids - self.seen_ids)        # someone new walked in

    def process(self, frame):
        track = self.hub.get("track")
        if track:
            self.seen_ids |= {p["id"] for p in (track.fresh_result() or [])}
        faces = self.fm.app.get(frame.image)
        if self.registering:
            self._take_sample(faces)

        out = []
        for f in faces:
            name, best = "unknown", -1.0
            for who, emb in self.known.items():
                s = float(np.max(emb @ f.normed_embedding))
                if s > best:
                    name, best = who, s
            if best < self.fm.THRESHOLD:
                name = "unknown"
            out.append({"name": name, "score": best, "box": tuple(int(v) for v in f.bbox)})
        return out

    def _take_sample(self, faces):
        if not faces:
            return
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        e = f.normed_embedding
        if not self.samples or max(float(np.dot(e, s)) for s in self.samples) < SAMPLE_DIFF:
            self.samples.append(e.copy())
        n = self.fm.NUM_SAMPLES
        self.message = f"registering {self.registering}: {len(self.samples)}/{n}"
        if len(self.samples) >= n:
            path = os.path.join(self.fm.FACES_DIR, f"{self.registering}.npy")
            tmp = path + ".tmp.npy"
            np.save(tmp, np.array(self.samples, dtype=np.float32))
            os.replace(tmp, path)
            print(f"[face] saved {n} samples for {self.registering}")
            self.registering = None
            self.reload_known()
