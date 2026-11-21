"""
display.py — draw every module's latest result onto one dashboard image,
and serve that image to browsers.

Compositor.render() fuses results per tracked person:
    name        -> the face module's track_names (it matches faces to tracks)
    expression  -> the person box containing the face centre
    gestures    -> the person box containing the shoulder midpoint
    distance    -> depth module, keyed by track ID
Without tracking, faces and gestures are still drawn on their own.

WebServer serves on http://<host>:<port>/
    /         page with the live view and module buttons
    /stream   MJPEG of the dashboard
    /status   JSON of module states and people
    /toggle?m=<key>, /idle (auto-idle on/off), /register?name=<name>, /cancel,
    /forget?name=<name>, /forget_all
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from workers import boxes_contain

try:
    import sysstats
except Exception as _e:                       # stats are optional — never break the demo
    sysstats = None
    print(f"[demo] system stats unavailable: {_e!r}")

_stats_warned = False


def _safe_system() -> dict:
    """Usage stats for the GUI. Any failure here must not take /status down."""
    global _stats_warned
    if sysstats is None:
        return {}
    try:
        return sysstats.snapshot()
    except Exception as e:
        if not _stats_warned:
            _stats_warned = True
            print(f"[demo] system stats failed (reported once): {e!r}", flush=True)
        return {}


def _json_default(o):
    """numpy scalars/arrays and anything else json can't encode natively."""
    if hasattr(o, "item") and getattr(o, "ndim", 0) == 0:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)

PANEL_W = 430
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, AMBER, RED = (240, 240, 240), (150, 150, 150), (90, 220, 90), (0, 190, 255), (70, 70, 230)
BG = (32, 28, 26)
TRACK_COLORS = [(255, 170, 60), (90, 220, 90), (60, 120, 255), (230, 90, 220),
                (40, 220, 230), (200, 200, 70), (130, 130, 255)]
STATE_COLORS = {"running": GREEN, "loading": AMBER, "off": GREY, "unavailable": RED, "error": RED}


def _text(img, s, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class Compositor:
    def __init__(self, source, workers: dict, presence=None):
        self.source = source
        self.w = workers
        self.presence = presence   # presence.Presence (auto-idle), or None
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
        faces = (fw.fresh_result(max_age_s=1.5) if fw else None) or []
        name_for = fw.name_for if fw and fw.state == "running" else (lambda tid: None)
        exprs = self._res("expr") or []
        gest = self._res("gesture")
        dist = self._res("depth") or {}
        people = []
        if tracks:
            # each expression belongs to one person: the smallest box holding the face centre
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
            expr_of = {}
            for e in exprs:
                cx, cy = (e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2
                holders = [t for t in tracks if boxes_contain(t["box"], cx, cy)]
                if holders:
                    expr_of.setdefault(min(holders, key=lambda t: area(t["box"]))["id"], e)
            # crop mode: gestures already carry their track ID
            by_track = {r["track"]: r["gestures"] for r in (gest or {}).get("people") or []}
            # smallest box first, so overlapping people get the most specific match
            for t in sorted(tracks, key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1])):
                p = {"id": t["id"], "box": t["box"], "name": name_for(t["id"]), "expr": None,
                     "gestures": [], "dist": dist.get(t["id"])}
                if t["id"] in expr_of:
                    p["expr"] = (expr_of[t["id"]]["emotion"], expr_of[t["id"]]["conf"])
                if by_track:
                    p["gestures"] = by_track.get(t["id"], [])
                elif gest and gest["anchor"] and boxes_contain(t["box"], *gest["anchor"]) \
                        and not any(q["gestures"] for q in people):
                    p["gestures"] = gest["gestures"]
                people.append(p)
            people.sort(key=lambda p: p["id"])
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
        bodies = (gest.get("people") or [gest]) if gest else []
        if gest and any(b["pose"] for b in bodies):
            g = self.w["gesture"]
            for body in bodies:                 # one skeleton per person in crop mode
                pts = body["pose"]
                if pts:
                    for a, b in g.pose_connections:
                        if pts[a][2] > 0.5 and pts[b][2] > 0.5:
                            cv2.line(img, pts[a][:2], pts[b][:2], (230, 230, 230), 2, cv2.LINE_AA)
                for hand in body["hands"]:
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
            _text(img, f"{f['name']} {f['score']:.2f}", (x1, max(y1 - 6, 12)), 0.45 * scale, color)
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
        y += 22
        if self.presence is not None:
            pr = self.presence
            color = {"idle": AMBER, "aware": (255, 200, 120)}.get(pr.mode, (120, 230, 120)) if pr.enabled else GREY
            _text(pn, pr.line()[:56], (18, y), 0.45, color)
            y += 22
        y += 12
        for wk in self.w.values():
            color = AMBER if wk.paused_for and wk.state == "running" else STATE_COLORS.get(wk.state, GREY)
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
            _text(pn, "Enter = start (only that person in view), Esc = cancel", (18, h - 40), 0.42, GREY)
        elif self.notice and time.time() - self.notice_t < 8:
            _text(pn, self.notice[:56], (18, h - 60), 0.45, AMBER)
        _text(pn, "1-5 toggle  i idle  r register  x cancel  f full  q quit", (18, h - 16), 0.42, GREY)

    def status(self) -> dict:
        fw = self.w.get("face")
        loaded = fw is not None and fw.state in ("running", "off", "error")
        return {
            "faces": {"known": sorted(fw.known) if loaded else [],
                      "registering": bool(loaded and fw.registering),
                      "message": (fw.register_status if loaded else "")},
            "camera": {"status": self.source.status, "fps": round(self.source.fps, 1)},
            "modules": {k: {"label": w.label,
                            "state": "paused" if w.paused_for and w.state == "running" else w.state,
                            "hz": round(w.hz, 1),
                            "ms": round(w.ms), "message": w.error or w.message} for k, w in self.w.items()},
            "people": [{k: v for k, v in p.items() if k != "box"} for p in self.people],
            "system": _safe_system(),
            "presence": self.presence.status() if self.presence is not None else None,
        }


# ── Web view ───────────────────────────────────────────────────────────────────

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLaDOS Vision</title>
<style>
:root{--bg:#1a1c20;--card:#2c3038;--line:#444;--fg:#eee;--dim:#9aa;--ok:#5c5;--bad:#c55}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header,section{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 16px}
h1{font-size:18px;margin:0 12px 0 0}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 12px;cursor:pointer;font:inherit}
button:active{transform:translateY(1px)}
button.on{border-color:var(--ok);color:#9f9}button.off,button.paused{color:var(--dim)}
button.unavailable,button.error{border-color:var(--bad);color:#f99}
button.small{padding:3px 8px;font-size:13px}
input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px;font:inherit}
#msg{color:#fc6}.dim{color:var(--dim)}
img{display:block;width:100%;height:auto}
</style></head><body>
<header><h1>GLaDOS Vision</h1><span id="mods"></span><button id="idleb" title="pause heavy features while nobody is close">Auto-idle</button><span id="mode" class="dim"></span></header>
<section><b>Faces</b>
<input id="nm" placeholder="name" size="10"><button id="regb">Register</button><button id="canb">Cancel</button>
<span class="dim">Known:</span><span id="known"></span><button class="small" id="fall">Forget all</button>
<span id="msg"></span></section>
<img src="/stream">
<script>
const $ = id => document.getElementById(id), btns = {};
async function call(url){try{const t=await (await fetch(url)).text(); if(t&&t!='ok')$('msg').textContent=t;}catch(e){$('msg').textContent='no connection';} poll(true);}
$('regb').onclick=()=>call('/register?name='+encodeURIComponent($('nm').value));
$('canb').onclick=()=>call('/cancel');
$('idleb').onclick=()=>call('/idle');
$('fall').onclick=()=>{if(confirm('Forget every registered face?'))call('/forget_all');};
let timer=null, knownKey='';
async function poll(once){
  try{const s=await (await fetch('/status',{cache:'no-store'})).json();
    for(const [k,v] of Object.entries(s.modules)){
      let b=btns[k];
      if(!b){b=btns[k]=document.createElement('button');b.onclick=()=>call('/toggle?m='+k);$('mods').appendChild(b);}
      b.textContent=v.label+(v.state=='running'?' · '+v.hz+' Hz':' ('+v.state+')');
      b.className=v.state=='running'?'on':v.state;b.title=v.message||'';
    }
    const key=s.faces.known.join(',');
    if(key!==knownKey){knownKey=key;const k=$('known');k.textContent='';
      if(!s.faces.known.length)k.textContent='none';
      for(const n of s.faces.known){const b=document.createElement('button');b.className='small';
        b.textContent=n+' ✕';b.title='forget '+n;b.onclick=()=>{if(confirm('Forget '+n+'?'))call('/forget?name='+encodeURIComponent(n));};k.appendChild(b);}}
    if(s.faces.message)$('msg').textContent=s.faces.message;
    $('canb').style.display=s.faces.registering?'':'none';
    if(s.presence){const p=s.presence,r=p.room||{};
      $('idleb').textContent='Auto-idle: '+(p.enabled?'on':'off');$('idleb').className=p.enabled?'on':'off';
      $('mode').textContent=(p.enabled?p.mode.toUpperCase()+' ('+p.reason+') · ':'')+'motion '+(r.optical_flow||'?')+
        ', light '+(r.light||'?')+(r.blocked?', CAMERA BLOCKED':'');}
  }catch(e){}
  if(!once){clearTimeout(timer);timer=setTimeout(poll,1000);}
}
poll();
</script></body></html>"""


class WebServer:
    def __init__(self, comp: Compositor, port: int, on_register):
        self.comp = comp
        self.on_register = on_register

        def face_call(method, *a):
            fw = comp.w.get("face")
            if fw is None or not hasattr(fw, method):
                return "face recognition is not running"
            msg = getattr(fw, method)(*a)
            comp.say(msg)
            return msg
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
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/":
                    self._send(PAGE, "text/html")
                elif u.path == "/status":
                    try:
                        body = json.dumps(server.comp.status(), default=_json_default)
                    except Exception as e:
                        # Reply with the reason instead of dropping the connection,
                        # so the Mac's VISION box shows what went wrong.
                        import traceback
                        traceback.print_exc()
                        body = json.dumps({"error": f"{type(e).__name__}: {e}", "people": []})
                    self._send(body, "application/json")
                elif u.path == "/toggle":
                    w = server.comp.w.get(q.get("m", [""])[0])
                    if w:
                        w.toggle()
                    self._send("ok")
                elif u.path == "/idle":
                    p = server.comp.presence
                    self._send(p.toggle() if p is not None else "auto-idle not available")
                elif u.path == "/register":
                    self._send(server.on_register(q.get("name", [""])[0]))
                elif u.path == "/cancel":
                    self._send(face_call("cancel_register"))
                elif u.path == "/forget":
                    self._send(face_call("forget", q.get("name", [""])[0]))
                elif u.path == "/forget_all":
                    self._send(face_call("forget_all"))
                elif u.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                with server.cond:
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
                    with server.cond:
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
