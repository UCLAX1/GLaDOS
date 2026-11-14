"""
display.py — draw every module's latest result onto one dashboard image,
and serve that image to browsers.

Compositor.render() fuses results per tracked person:
    face name / expression  -> the person box containing the face centre
    gestures                -> the person box containing the shoulder midpoint
    distance                -> depth module, keyed by track ID
A name sticks to its track ID once recognised, so it stays while the person
looks away. Without tracking, faces and gestures are still drawn on their own.

WebServer serves on http://<host>:<port>/
    /         page with the live view and module buttons
    /stream   MJPEG of the dashboard
    /status   JSON of module states and people
    /toggle?m=<key>, /register?name=<name>
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from workers import boxes_contain

PANEL_W = 430
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, AMBER, RED = (240, 240, 240), (150, 150, 150), (90, 220, 90), (0, 190, 255), (70, 70, 230)
BG = (32, 28, 26)
TRACK_COLORS = [(255, 170, 60), (90, 220, 90), (60, 120, 255), (230, 90, 220),
                (40, 220, 230), (200, 200, 70), (130, 130, 255)]
STATE_COLORS = {"running": GREEN, "loading": AMBER, "off": GREY, "unavailable": RED, "error": RED}
NAME_HOLD_S = 30          # forget a track's name if its ID hasn't been seen for this long


def _text(img, s, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class Compositor:
    def __init__(self, source, workers: dict):
        self.source = source
        self.w = workers
        self.names = {}            # track id -> (name, last seen time)
        self.people = []           # last fused list, for /status
        self.notice = ""           # one-line message at the bottom of the panel
        self.notice_t = 0.0
        self.typing = None         # name being typed for registration, or None
        self.fps = 0.0
        self._last = 0.0

    def say(self, msg):
        self.notice, self.notice_t = msg, time.time()
        print(f"[demo] {msg}")

    def _res(self, key):
        wk = self.w.get(key)
        return wk.fresh_result() if wk else None

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(self):
        tracks = self._res("track")
        fw = self.w.get("face")
        faces = (fw.fresh_result(max_age_s=4.0) if fw else None) or []
        exprs = self._res("expr") or []
        gest = self._res("gesture")
        dist = self._res("depth") or {}
        now = time.time()
        people = []
        if tracks:
            # smallest box first, so overlapping people get the most specific match
            for t in sorted(tracks, key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1])):
                p = {"id": t["id"], "box": t["box"], "name": None, "expr": None,
                     "gestures": [], "dist": dist.get(t["id"])}
                for f in faces:
                    cx, cy = (f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2
                    if boxes_contain(t["box"], cx, cy) and f["name"] != "unknown":
                        self.names[t["id"]] = (f["name"], now)
                for e in exprs:
                    cx, cy = (e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2
                    if p["expr"] is None and boxes_contain(t["box"], cx, cy):
                        p["expr"] = (e["emotion"], e["conf"])
                if gest and gest["anchor"] and boxes_contain(t["box"], *gest["anchor"]) \
                        and not any(q["gestures"] for q in people):
                    p["gestures"] = gest["gestures"]
                if t["id"] in self.names:
                    p["name"] = self.names[t["id"]][0]
                    self.names[t["id"]] = (p["name"], now)
                people.append(p)
            people.sort(key=lambda p: p["id"])
        self.names = {k: v for k, v in self.names.items() if now - v[1] < NAME_HOLD_S}
        self.people = people
        return people, faces, exprs, gest

    # ── Drawing ───────────────────────────────────────────────────────────────

    def render(self):
        frame = self.source.latest()
        if frame is None:
            img = np.zeros((720, 1280, 3), np.uint8)
            _text(img, f"Waiting for camera: {self.source.status}", (40, 360), 0.9, AMBER, 2)
        else:
            img = frame.image.copy()
        people, faces, exprs, gest = self.fuse()
        if frame is not None:
            self._draw_video(img, people, faces, exprs, gest)
        h = img.shape[0]
        panel = np.full((h, PANEL_W, 3), BG, np.uint8)
        self._draw_panel(panel, people)

        now = time.time()
        if self._last:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last, 1e-6)
        self._last = now
        return np.hstack([img, panel])

    def _draw_video(self, img, people, faces, exprs, gest):
        scale = max(img.shape[0] / 720, 0.5)
        if gest and gest["pose"]:
            g = self.w["gesture"]
            pts = gest["pose"]
            for a, b in g.pose_connections:
                if pts[a][2] > 0.5 and pts[b][2] > 0.5:
                    cv2.line(img, pts[a][:2], pts[b][:2], (230, 230, 230), 2, cv2.LINE_AA)
            for hand in gest["hands"]:
                for a, b in g.hand_connections:
                    cv2.line(img, hand[a], hand[b], (60, 200, 255), 2, cv2.LINE_AA)
        elif gest:
            for hand in gest["hands"]:
                for x, y in hand:
                    cv2.circle(img, (x, y), 3, (60, 200, 255), -1)

        for f in faces:
            x1, y1, x2, y2 = f["box"]
            color = GREEN if f["name"] != "unknown" else AMBER
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        for e in exprs:
            x1, y1, x2, y2 = e["box"]
            _text(img, f"{e['emotion']} {e['conf']:.0%}", (x1, y2 + int(20 * scale)), 0.55 * scale, (255, 220, 120))

        for p in people:
            x1, y1, x2, y2 = p["box"]
            color = TRACK_COLORS[p["id"] % len(TRACK_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"ID {p['id']}" + (f"  {p['name']}" if p["name"] else "")
            if p["dist"] is not None:
                label += f"  {p['dist']:.1f} m"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.75 * scale, 2)
            cv2.rectangle(img, (x1, max(y1 - th - 14, 0)), (x1 + tw + 12, max(y1, th + 14)), color, -1)
            cv2.putText(img, label, (x1 + 6, max(y1 - 8, th + 6)), FONT, 0.75 * scale, (0, 0, 0), 2, cv2.LINE_AA)
            for i, gs in enumerate(p["gestures"]):
                _text(img, gs, (x1 + 8, y1 + int((32 + 30 * i) * scale)), 0.8 * scale, (80, 255, 255), 2)

        # gestures with no tracked person to attach to (tracking off): top-left banner
        if gest and gest["gestures"] and not any(p["gestures"] for p in people):
            _text(img, " | ".join(gest["gestures"]), (20, int(45 * scale)), 1.0 * scale, (80, 255, 255), 2)

    def _draw_panel(self, pn, people):
        y = 40
        _text(pn, "GLaDOS Vision", (18, y), 0.9, WHITE, 2)
        y += 30
        _text(pn, f"camera: {self.source.status}"[:44], (18, y), 0.45, GREY)
        y += 22
        _text(pn, f"camera {self.source.fps:4.1f} fps   display {self.fps:4.1f} fps", (18, y), 0.45, GREY)
        y += 34
        for wk in self.w.values():
            color = STATE_COLORS.get(wk.state, GREY)
            cv2.circle(pn, (26, y - 5), 6, color, -1)
            _text(pn, f"[{wk.hotkey}] {wk.label}", (40, y), 0.55, WHITE)
            y += 20
            _text(pn, wk.status_line()[:52], (40, y), 0.42, color if wk.state != "running" else GREY)
            y += 28
        y += 6
        cv2.line(pn, (18, y), (PANEL_W - 18, y), (80, 80, 80), 1)
        y += 30
        _text(pn, f"People: {len(people)}", (18, y), 0.65, WHITE, 2)
        y += 30
        for p in people[:6]:
            color = TRACK_COLORS[p["id"] % len(TRACK_COLORS)]
            cv2.rectangle(pn, (18, y - 14), (28, y - 4), color, -1)
            head = f"ID {p['id']}  {p['name'] or '?'}"
            if p["dist"] is not None:
                head += f"   {p['dist']:.1f} m"
            _text(pn, head, (36, y), 0.55, WHITE)
            y += 22
            detail = []
            if p["expr"]:
                detail.append(f"{p['expr'][0]} {p['expr'][1]:.0%}")
            detail += p["gestures"]
            if detail:
                _text(pn, "  ".join(detail)[:52], (36, y), 0.45, (80, 255, 255))
                y += 22
            y += 8
        h = pn.shape[0]
        if self.typing is not None:
            cv2.rectangle(pn, (12, h - 96), (PANEL_W - 12, h - 56), (70, 60, 50), -1)
            _text(pn, f"Name: {self.typing}_", (22, h - 68), 0.65, WHITE, 2)
            _text(pn, "Enter = register largest face, Esc = cancel", (18, h - 40), 0.42, GREY)
        elif self.notice and time.time() - self.notice_t < 8:
            _text(pn, self.notice[:56], (18, h - 60), 0.45, AMBER)
        _text(pn, "1-5 toggle  r register face  f fullscreen  q quit", (18, h - 16), 0.42, GREY)

    def status(self) -> dict:
        return {
            "camera": {"status": self.source.status, "fps": round(self.source.fps, 1)},
            "modules": {k: {"label": w.label, "state": w.state, "hz": round(w.hz, 1),
                            "ms": round(w.ms), "message": w.error or w.message} for k, w in self.w.items()},
            "people": [{k: v for k, v in p.items() if k != "box"} for p in self.people],
        }


# ── Web view ───────────────────────────────────────────────────────────────────

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLaDOS Vision</title>
<style>
body{margin:0;background:#1a1c20;color:#eee;font:15px system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 16px}
h1{font-size:18px;margin:0 12px 0 0}
button{background:#2c3038;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
button.on{border-color:#5c5;color:#9f9}button.unavailable,button.error{border-color:#c55;color:#f99}
input{background:#2c3038;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
img{display:block;width:100%;height:auto}
</style></head><body>
<header><h1>GLaDOS Vision</h1><span id="mods"></span>
<input id="nm" placeholder="name" size="10"><button onclick="reg()">Register face</button><span id="msg"></span></header>
<img src="/stream">
<script>
async function poll(){try{const s=await (await fetch('/status')).json();
const m=document.getElementById('mods');m.innerHTML='';
for(const [k,v] of Object.entries(s.modules)){const b=document.createElement('button');
b.textContent=v.label+(v.state=='running'?' '+v.hz+' Hz':' ('+v.state+')');
b.className=v.state=='running'?'on':v.state;b.title=v.message;b.onclick=()=>fetch('/toggle?m='+k);m.appendChild(b);}
}catch(e){} setTimeout(poll,1000)}
async function reg(){const n=document.getElementById('nm').value;
document.getElementById('msg').textContent=await (await fetch('/register?name='+encodeURIComponent(n))).text();}
poll();
</script></body></html>"""


class WebServer:
    def __init__(self, comp: Compositor, port: int, on_register):
        self.comp = comp
        self.on_register = on_register
        self.jpeg = None
        self.clients = 0
        self.cond = threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype="text/plain"):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/":
                    self._send(PAGE, "text/html")
                elif u.path == "/status":
                    self._send(json.dumps(server.comp.status()), "application/json")
                elif u.path == "/toggle":
                    w = server.comp.w.get(q.get("m", [""])[0])
                    if w:
                        w.toggle()
                    self._send("ok")
                elif u.path == "/register":
                    self._send(server.on_register(q.get("name", [""])[0]))
                elif u.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                server.clients += 1
                try:
                    last = None
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.jpeg is not last, timeout=2)
                            last = server.jpeg
                        if last is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(last)).encode() + b"\r\n\r\n" + last + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    server.clients -= 1

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, name="web", daemon=True).start()

    def publish(self, img):
        """Encode the dashboard for browsers, only while someone is watching."""
        if self.clients <= 0:
            return
        h, w = img.shape[:2]
        if w > 1600:                                   # keep Wi-Fi bandwidth sane
            img = cv2.resize(img, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def close(self):
        self.httpd.shutdown()
