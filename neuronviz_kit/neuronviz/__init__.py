"""neuronviz: watch a small PyTorch model learn, live, in a browser tab beside PyCharm.

    import neuronviz as nv
    model = nn.Sequential(...)
    nv.attach(model, X, y)        # one line, after building the model and data
    ... your normal training loop, untouched ...

Every few optimizer steps the model is snapshotted (decision map, weights, loss,
accuracy) and streamed to http://127.0.0.1:8765. Keep that tab open: it
reconnects and resets on every run, and you can scrub back through training.
"""
import atexit
import json
import os
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

__all__ = ["attach"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PORT = 8765
_MAX_FRAMES = 300
_hub = None


class _Hub:
    """Shared state between the training thread and the HTTP thread."""

    def __init__(self):
        self.run_id = uuid.uuid4().hex[:8]
        self.meta = {}
        self.frames = []
        self.done = False
        self.last_poll = 0.0
        self.lock = threading.Lock()
        self.port = None


def _handler(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/state":
                q = parse_qs(url.query)
                hub.last_poll = time.time()
                after = int(q.get("after", ["-1"])[0])
                if q.get("run", [""])[0] != hub.run_id:
                    after = -1  # a new run: send everything
                with hub.lock:
                    out = {
                        "run": hub.run_id,
                        "meta": hub.meta,
                        "done": hub.done,
                        "frames": [f for f in hub.frames if f["step"] > after],
                    }
                self._send(json.dumps(out).encode(), "application/json")
            else:
                with open(os.path.join(_HERE, "index.html"), "rb") as fh:
                    self._send(fh.read(), "text/html; charset=utf-8")

    return Handler


def _start_server():
    global _hub
    if _hub is not None:
        return _hub
    hub = _Hub()
    for port in range(_BASE_PORT, _BASE_PORT + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), _handler(hub))
        except OSError:
            continue
        hub.port = port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _hub = hub
        return hub
    raise RuntimeError("neuronviz: no free port between 8765 and 8774")


class _Viz:
    def __init__(self, model, X, y, every, loss_fn, slow, grid, open_browser):
        self.model, self.every, self.loss_fn, self.slow, self.G = model, max(1, every), loss_fn, slow, grid
        self.dev = next(model.parameters()).device
        self.X = torch.as_tensor(X).float().to(self.dev)
        y = torch.as_tensor(y).to(self.dev)

        leaves = [m for m in model.modules() if not list(m.children())]
        self.prob_out = isinstance(leaves[-1], nn.Sigmoid)
        layers = []
        for m in leaves:
            if isinstance(m, nn.Linear):
                if not layers:
                    layers.append({"n": m.in_features, "act": None})
                layers.append({"n": m.out_features, "act": None, "_m": m})
            elif layers and type(m).__module__.endswith("activation"):
                layers[-1]["act"] = type(m).__name__
        if len(layers) < 2:
            raise ValueError("neuronviz draws models built from nn.Linear layers; none found.")
        self.linears = [l.pop("_m") for l in layers[1:]]

        d_out = layers[-1]["n"]
        if d_out > 1:
            self.mode, self.y = "multi", y.long()
        elif set(torch.unique(y).tolist()) <= {0, 1}:
            self.mode, self.y = "bin", y.float()
        else:
            self.mode, self.y = "reg", y.float()
        self.ymin, self.ymax = float(self.y.min()), float(self.y.max())

        self.has_map = self.X.shape[1] == 2
        lo, hi = self.X.min(0).values, self.X.max(0).values
        pad = (hi - lo) * 0.12
        self.bounds = [float(lo[0] - pad[0]), float(hi[0] + pad[0]), float(lo[1] - pad[1]), float(hi[1] + pad[1])] if self.has_map else [0, 1, 0, 1]

        n = min(len(self.X), 800)
        pts = self.X[:n].cpu().tolist() if self.has_map else []
        lab = self.y[:n].cpu().tolist()
        meta = {
            "layers": layers,
            "mode": self.mode,
            "grid": grid,
            "has_map": self.has_map,
            "bounds": self.bounds,
            "yrange": [self.ymin, self.ymax],
            "points": [[round(p[0], 3), round(p[1], 3), round(l, 3)] for p, l in zip(pts, lab)],
            "params": sum(p.numel() for p in model.parameters()),
            "n": len(self.X),
            "summary": " \u2192 ".join(str(l["n"]) for l in layers),
        }

        self.hub = _start_server()
        with self.hub.lock:
            self.hub.run_id = uuid.uuid4().hex[:8]
            self.hub.meta, self.hub.frames, self.hub.done = meta, [], False
        self.count = 0
        self._ids = {id(p) for p in model.parameters()}
        self._snapshot(0)
        self._hook = register_optimizer_step_post_hook(self._on_step)
        atexit.register(self._finish)

        url = f"http://127.0.0.1:{self.hub.port}"
        print(f"neuronviz: live view at {url}")
        if open_browser:
            t0 = time.time()
            while time.time() - t0 < 1.0 and self.hub.last_poll == 0:
                time.sleep(0.05)
            if self.hub.last_poll == 0:  # no tab is listening yet
                webbrowser.open(url)

    # -- optimizer hook -------------------------------------------------
    def _on_step(self, opt, args, kwargs):
        if not any(id(p) in self._ids for g in opt.param_groups for p in g["params"]):
            return
        self.count += 1
        if self.count % self.every == 0:
            self._snapshot(self.count)
            if self.slow:
                time.sleep(self.slow)

    def _finish(self):
        if self.count and not self.hub.frames[-1]["step"] == self.count:
            self._snapshot(self.count)
        self.hub.done = True
        time.sleep(0.8)  # give the open tab time to fetch the last frames

    # -- snapshot -------------------------------------------------------
    def _loss_acc(self, out):
        y = self.y
        if self.loss_fn is not None:
            try:
                loss = self.loss_fn(out, y)
            except Exception:
                loss = self.loss_fn(out.squeeze(-1), y)
        if self.mode == "bin":
            o = out.reshape(-1)
            p = o if self.prob_out else torch.sigmoid(o)
            if self.loss_fn is None:
                loss = F.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), y)
            return float(loss), float(((p > 0.5) == (y > 0.5)).float().mean())
        if self.mode == "multi":
            if self.loss_fn is None:
                loss = F.cross_entropy(out, y)
            return float(loss), float((out.argmax(-1) == y).float().mean())
        if self.loss_fn is None:
            loss = F.mse_loss(out.reshape(-1), y)
        return float(loss), None

    def _grid_hex(self):
        x0, x1, y0, y1 = self.bounds
        G = self.G
        gx = torch.linspace(x0, x1, G, device=self.dev)
        gy = torch.linspace(y0, y1, G, device=self.dev)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")  # row = y, column = x
        o = self.model(torch.stack([xx, yy], -1).reshape(-1, 2))
        if self.mode == "bin":
            o = o.reshape(-1)
            p = o if self.prob_out else torch.sigmoid(o)
            return bytes((p.clamp(0, 1) * 255).round().byte().tolist()).hex()
        if self.mode == "multi":
            conf, cls = torch.softmax(o, -1).max(-1)
            both = torch.stack([cls.clamp(max=255), (conf * 255).round()], -1).byte()
            return bytes(both.reshape(-1).tolist()).hex()
        span = (self.ymax - self.ymin) or 1.0
        t = ((o.reshape(-1) - self.ymin) / span).clamp(0, 1)
        return bytes((t * 255).round().byte().tolist()).hex()

    def _snapshot(self, step):
        m = self.model
        was_training = m.training
        m.eval()
        try:
            with torch.no_grad():
                loss, acc = self._loss_acc(m(self.X))
                grid = self._grid_hex() if self.has_map else ""
                weights = [[[round(v, 3) for v in row[:16]] for row in l.weight[:16].tolist()] for l in self.linears]
        finally:
            m.train(was_training)
        frame = {"step": step, "loss": loss, "acc": acc, "grid": grid, "w": weights}
        with self.hub.lock:
            self.hub.frames.append(frame)
            if len(self.hub.frames) > _MAX_FRAMES:  # thin out, keep the whole run
                self.hub.frames[:] = self.hub.frames[::2]
                self.every *= 2


def attach(model, X, y, every=10, loss_fn=None, slow=0.05, grid=64, open_browser=True):
    """Start watching `model`.

    X, y        the data to draw (tensors or numpy). 2 input features gives a decision map.
    every       snapshot every N optimizer steps (adapts upward on long runs).
    loss_fn     only if your loss isn't BCE / cross-entropy / MSE; called as loss_fn(output, y).
    slow        seconds to pause after each snapshot so fast runs are watchable; 0 disables.
    """
    return _Viz(model, X, y, every, loss_fn, slow, grid, open_browser)
