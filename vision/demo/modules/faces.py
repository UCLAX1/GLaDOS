"""
faces.py — who is it? Face recognition from vision/face_memory.py.

Uses that script's InsightFace model (buffalo_l), match threshold and
known_faces/*.npy format. Changes for running alongside everything else:
  * known faces are loaded once (and after a change), not every frame
  * it only runs when needed: a new tracked person appeared, or every PERIOD_S
  * names are attached to tracked people here, once per recognition, so the
    display never re-applies an old result to boxes that have moved

Naming people (track_names):
  * each face goes to one tracked person: the smallest box holding its centre
  * a name belongs to one person at a time: the best match keeps it
  * recognising someone as "unknown" clears a wrong name; a name only stays
    while the face can't be seen (person turned away), up to NAME_HOLD_S

Registration (register / cancel_register / forget / forget_all):
  * samples are only taken while exactly ONE face is visible, as in
    face_memory.register_person(), so a bystander can't be recorded
  * every sample must match the earlier ones (same person throughout)
  * refused if the face already belongs to someone else (shows who)
  * gives up after REGISTER_TIMEOUT_S

Result: [{"name": "arjun" | "unknown", "score": 0.62, "box": (x1, y1, x2, y2), "track": 3 | None}]
"""

import os
import re
import threading
import time

import numpy as np

from workers import JETSON, Worker, boxes_contain, onnx_threads

PERIOD_S = float(os.environ.get("DEMO_FACE_PERIOD", "1.0"))
SAMPLE_DIFF = 0.995        # skip samples this similar to one already taken (as face_memory)
SAME_PERSON = 0.30         # a new sample must be at least this similar to the earlier ones
REGISTER_TIMEOUT_S = 90         # ~15 samples at the Jetson's ~1-4 s per face pass
NAME_HOLD_S = 30           # keep a name for a track this long after its face was last seen
UNKNOWN_CLEARS = 2         # this many "unknown" results in a row clear a track's name


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


class FaceWorker(Worker):
    key, label, hotkey = "face", "Face recognition", "3"
    # defaults so the panel / browser can ask before load() has run
    known = {}
    registering = None
    register_status = ""
    track_names = {}

    def load(self):
        with onnx_threads(2 if JETSON else 0):
            import face_memory as fm             # builds buffalo_l on import
        # buffalo_l also runs 2 landmark models and age/gender on every face;
        # recognition only needs the detector + embedding (~2x faster without them)
        fm.app.models = {k: m for k, m in fm.app.models.items() if k in ("detection", "recognition")}
        self.fm = fm
        self.lock = threading.Lock()
        self.known = {}
        self.seen_ids = set()
        self.track_names = {}                    # track id -> {"name", "score", "t"}
        self.registering = None                  # name being registered
        self.register_status = ""                # shown in the panel / browser
        self.samples = []
        self.register_t = 0.0
        self.reload_known()

    # ── Known faces ───────────────────────────────────────────────────────────

    def reload_known(self):
        known = {}
        for f in sorted(os.listdir(self.fm.FACES_DIR)):
            if f.endswith(".npy"):
                e = np.load(os.path.join(self.fm.FACES_DIR, f), allow_pickle=False)
                known[os.path.splitext(f)[0]] = e.reshape(1, -1) if e.ndim == 1 else e
        with self.lock:
            self.known = known
            # drop names that no longer exist
            self.track_names = {k: v for k, v in self.track_names.items() if v["name"] in known}
        self.message = f"{len(known)} known: {', '.join(known) or 'none'}"

    def forget(self, name: str) -> str:
        path = os.path.join(self.fm.FACES_DIR, f"{name}.npy")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not os.path.exists(path):
            return f"no registered face called {name!r}"
        os.remove(path)
        self.reload_known()
        return f"forgot {name}"

    def forget_all(self) -> str:
        names = list(self.known)
        for n in names:
            os.remove(os.path.join(self.fm.FACES_DIR, f"{n}.npy"))
        self.reload_known()
        return f"forgot {len(names)} face(s)"

    def best_match(self, emb):
        name, best = "unknown", -1.0
        for who, e in self.known.items():
            s = float(np.max(e @ emb))
            if s > best:
                name, best = who, s
        return (name if best >= self.fm.THRESHOLD else "unknown"), best

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str) -> str:
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            return "name must be letters, numbers, _ or -"
        if self.state != "running":
            return f"face recognition is {self.state}"
        self.samples, self.registering, self.register_t = [], name, time.time()
        self.register_status = f"registering {name}: only {name} in view, slowly turn your head"
        return self.register_status

    def cancel_register(self) -> str:
        if not self.registering:
            return "not registering"
        name, self.registering = self.registering, None
        self.register_status = f"registration of {name} cancelled"
        return self.register_status

    def _take_sample(self, faces):
        name, n = self.registering, self.fm.NUM_SAMPLES
        if time.time() - self.register_t > REGISTER_TIMEOUT_S:
            self.registering = None
            self.register_status = f"registration of {name} timed out ({len(self.samples)}/{n}); try again"
            return
        if len(faces) != 1:
            self.register_status = (f"registering {name} {len(self.samples)}/{n}: paused, "
                                    f"{len(faces)} faces in view (need exactly 1)")
            return
        e = faces[0].normed_embedding
        other, score = self.best_match(e)
        if other not in ("unknown", name):
            self.registering = None
            self.register_status = f"not saved: that face is already registered as {other} ({score:.2f})"
            return
        if self.samples and float(np.mean([np.dot(e, s) for s in self.samples])) < SAME_PERSON:
            self.register_status = f"registering {name} {len(self.samples)}/{n}: skipped a sample that looks like someone else"
            return
        if not self.samples or max(float(np.dot(e, s)) for s in self.samples) < SAMPLE_DIFF:
            self.samples.append(e.copy())
        self.register_status = f"registering {name}: {len(self.samples)}/{n}, keep turning your head slowly"
        if len(self.samples) >= n:
            path = os.path.join(self.fm.FACES_DIR, f"{name}.npy")
            tmp = path + ".tmp.npy"
            np.save(tmp, np.array(self.samples, dtype=np.float32))
            os.replace(tmp, path)
            print(f"[face] saved {n} samples for {name}")
            self.registering = None
            self.register_status = f"registered {name}"
            self.reload_known()

    # ── Per frame ─────────────────────────────────────────────────────────────

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
        tracks = (track.fresh_result(max_age_s=0.5) if track else None) or []
        self.seen_ids |= {t["id"] for t in tracks}
        faces = self.fm.app.get(frame.image)
        if self.registering:
            self._take_sample(faces)

        out = []
        for f in faces:
            name, score = self.best_match(f.normed_embedding)
            box = tuple(int(v) for v in f.bbox)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            holders = [t for t in tracks if boxes_contain(t["box"], cx, cy)]
            tid = min(holders, key=lambda t: _area(t["box"]))["id"] if holders else None
            out.append({"name": name, "score": score, "box": box, "track": tid})
        self._update_names(out)
        return out

    def _update_names(self, faces):
        now = time.time()
        with self.lock:
            names = self.track_names
            # one face per track: keep its best-scoring face
            per_track = {}
            for f in faces:
                if f["track"] is not None and (f["track"] not in per_track
                                               or f["score"] > per_track[f["track"]]["score"]):
                    per_track[f["track"]] = f
            # one track per name: the best score this round wins
            winners = {}
            for tid, f in per_track.items():
                if f["name"] != "unknown" and (f["name"] not in winners
                                               or f["score"] > per_track[winners[f["name"]]]["score"]):
                    winners[f["name"]] = tid
            for tid, f in per_track.items():
                if f["name"] != "unknown" and winners[f["name"]] == tid:
                    for other in [k for k, v in names.items() if v["name"] == f["name"] and k != tid]:
                        del names[other]           # the name moved to this person
                    names[tid] = {"name": f["name"], "score": f["score"], "t": now, "misses": 0}
                elif f["name"] != "unknown":
                    names.pop(tid, None)           # someone else matched this name better
                elif tid in names:
                    # face visible but not recognised: one miss can be a bad angle,
                    # two in a row means the name was wrong (or someone swapped in)
                    names[tid]["misses"] += 1
                    if names[tid]["misses"] >= UNKNOWN_CLEARS:
                        del names[tid]
            for tid in [k for k, v in names.items() if now - v["t"] > NAME_HOLD_S]:
                del names[tid]

    def name_for(self, track_id):
        if self.state == "loading":
            return None
        with self.lock:
            v = self.track_names.get(track_id)
        return v["name"] if v else None

    def status_line(self):
        s = super().status_line()
        return self.register_status if self.registering else s
