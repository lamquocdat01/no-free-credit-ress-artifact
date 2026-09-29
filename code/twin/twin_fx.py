"""P1b' C1/C2 — appearance effects for night and turbulence twins (applied to the sprite layer).

Per-scene statistics are estimated from the scene's OWN real background frames (the frames used for the
plate, i.e. GT-empty frames):
  noise      : Poisson-Gaussian model  var(I) = a*I + b  fitted on (frame - plate) residuals binned by I.
  turbulence : dense optical flow (Farneback) between consecutive background frames -> displacement std
               sigma_f (px), spatial correlation length ell (px, 1/e of the flow autocorrelation along x) and
               temporal correlation rho (successive flow fields).
Steps (profile letters):
  a  per-channel histogram matching of the sprite to the ring of background around the box (quantile map;
     replaces the frame-level gain) -> gamma/contrast and colour cast follow the local background.
  b  vehicles only: headlights (moving towards the camera / sideways) or tail-lights (moving away) as
     gaussian spots at the lamp positions implied by the motion direction (from the GT trajectory),
     + bloom + halo; body darkened with the background darkness; motion blur along the box velocity.
  c  sensor noise from the fitted Poisson-Gaussian model on the sprite pixels + desaturation with darkness.
  t1 turbulence: spatially correlated random displacement field (gaussian-filtered white noise, std sigma_f,
     correlation ell), AR(1) in time with rho, applied to the sprite layer (rgb + alpha) by remapping.
  t2 t1 + time-varying gaussian blur of the sprite layer, sigma ~ U(0.5, 1.5) * 0.5 * sigma_f
     (heuristic tie of blur to the measured shimmer amplitude; documented, not fitted).
"""
from __future__ import annotations

import cv2
import numpy as np

L_W = np.array([0.114, 0.587, 0.299], np.float32)      # BGR luminance weights


def lum(img):
    return img[..., :3].astype(np.float32) @ L_W


# ───────────────────────────── scene statistics ─────────────────────────────
def scene_stats(imgs, plate):
    out = {}
    P = plate.astype(np.float32)
    Lp = lum(P)
    res, ints = [], []
    for im in imgs[: min(len(imgs), 40)]:
        r = im.astype(np.float32) - P
        res.append(r.reshape(-1, 3))
        ints.append(np.repeat(Lp.reshape(-1, 1), 3, 1))
    R = np.concatenate(res).ravel()
    I = np.concatenate(ints).ravel()
    bins = np.linspace(0, 255, 18)
    xs, vs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (I >= lo) & (I < hi)
        if m.sum() > 500:
            rr = R[m]
            q = np.percentile(np.abs(rr), 95)
            rr = rr[np.abs(rr) <= q]                       # robust: drop object/shadow outliers
            xs.append((lo + hi) / 2)
            vs.append(rr.var())
    if len(xs) >= 2:
        a, b = np.polyfit(xs, vs, 1)
    else:
        a, b = 0.0, float(np.mean(vs)) if vs else 1.0
    out["noise_a"], out["noise_b"] = float(max(a, 0.0)), float(max(b, 0.1))
    # turbulence / shimmer from consecutive background frames
    g = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs[: min(len(imgs), 30)]]
    flows = [cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(g) - 1)]
    if flows:
        fx = np.stack([f[..., 0] for f in flows])
        sig = float(np.median([f.std() for f in fx]))
        acs = []
        for f in fx[:10]:
            f0 = f - f.mean()
            den = (f0 * f0).mean() + 1e-9
            ac = [1.0] + [float((f0[:, :-k] * f0[:, k:]).mean() / den) for k in range(1, min(40, f.shape[1] // 2))]
            k = next((i for i, v in enumerate(ac) if v < np.exp(-1)), len(ac))
            acs.append(k)
        rho = float(np.median([np.corrcoef(fx[i].ravel(), fx[i + 1].ravel())[0, 1] for i in range(len(fx) - 1)])) if len(fx) > 1 else 0.0
        out.update(turb_sigma=sig, turb_ell=float(max(1, np.median(acs))), turb_rho=float(np.clip(rho, 0, 0.99)))
    else:
        out.update(turb_sigma=0.0, turb_ell=1.0, turb_rho=0.0)
    out["plate_lum_median"] = float(np.median(Lp))
    return out


# ───────────────────────────── helpers ─────────────────────────────
def ring_pixels(base, bbox):
    H, W = base.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = max(2, x1 - x0), max(2, y1 - y0)
    r = int(0.3 * max(bw, bh)) + 4
    rx0, ry0, rx1, ry1 = max(0, x0 - r), max(0, y0 - r), min(W, x1 + r), min(H, y1 + r)
    reg = base[ry0:ry1, rx0:rx1].astype(np.float32)
    inner = np.zeros(reg.shape[:2], bool)
    inner[max(0, y0 - ry0):y1 - ry0, max(0, x0 - rx0):x1 - rx0] = True
    px = reg[~inner] if (~inner).sum() > 10 else reg.reshape(-1, 3)
    return px


def hist_match(rgb, a, ring):
    """Per-channel quantile mapping of the sprite pixels (alpha > 0.5) onto the ring of background around
    the box (brightness, contrast AND colour cast, e.g. sodium street light)."""
    m = a > 0.5
    if m.sum() < 5 or len(ring) < 10:
        return rgb
    q = np.linspace(0, 1, 51)
    out = rgb.copy()
    for c in range(3):
        qs = np.quantile(rgb[..., c][m], q)
        qs = np.maximum.accumulate(qs + np.arange(len(qs)) * 1e-6)
        qr = np.quantile(ring[:, c], q)
        out[..., c] = np.interp(rgb[..., c], qs, qr)
    return np.clip(out, 0, 255)


def _premul_filter(layer, fn):
    """Apply a spatial filter to an RGBA layer in premultiplied space (no dark fringes from transparent px)."""
    a = layer[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([layer[..., :3].astype(np.float32) * a, a * 255.0], axis=2)
    pm = fn(pm)
    a2 = np.clip(pm[..., 3:4] / 255.0, 0, 1)
    rgb = np.where(a2 > 1e-3, pm[..., :3] / np.maximum(a2, 1e-3), 0)
    return np.concatenate([np.clip(rgb, 0, 255), a2 * 255.0], axis=2).astype(layer.dtype)


def motion_blur(layer, vx, vy):
    L = float(np.hypot(vx, vy))
    if L < 2:
        return layer
    n = int(min(25, round(L))) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    ang = np.arctan2(vy, vx)
    for t in np.linspace(-c, c, 2 * n):
        x, y = int(round(c + t * np.cos(ang))), int(round(c + t * np.sin(ang)))
        k[y, x] = 1
    k /= k.sum()
    return _premul_filter(layer, lambda x: cv2.filter2D(x, -1, k, borderType=cv2.BORDER_CONSTANT))


def lamp_layer(shape_hw, tw, th, px0, py0, vx, vy, dark):
    """Additive BGR light layer (full frame): lamps at the positions implied by the motion direction."""
    H, W = shape_hw
    lay = np.zeros((H, W, 3), np.float32)
    if dark < 0.2:
        return lay
    r = max(1.0, 0.06 * tw)
    if abs(vx) > abs(vy):
        fx = 0.93 if vx > 0 else 0.07
        pts = [(fx, 0.62)]
        col = np.array([220, 240, 255], np.float32)                          # warm white (BGR)
    elif vy >= 0:                                                            # moving down the image: towards camera
        pts = [(0.2, 0.66), (0.8, 0.66)]
        col = np.array([220, 240, 255], np.float32)
    else:                                                                    # moving away: tail-lights
        pts = [(0.2, 0.55), (0.8, 0.55)]
        col = np.array([40, 40, 255], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for u, v in pts:
        cx, cy = px0 + u * tw, py0 + v * th
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        core = 255.0 * np.exp(-d2 / (2 * r ** 2))
        bloom = 120.0 * dark * np.exp(-d2 / (2 * (3 * r) ** 2))
        halo = 40.0 * dark * np.exp(-d2 / (2 * (8 * r) ** 2))
        lay += (core + bloom + halo)[..., None] * (col / 255.0)[None, None, :]
    return lay


class TurbField:
    """AR(1) in time, spatially correlated displacement field for one event's sprite layer."""

    def __init__(self, sigma, ell, rho, rng):
        self.s, self.ell, self.rho, self.rng = sigma, ell, rho, rng
        self.fx = self.fy = None

    def step(self, h, w):
        def draw():
            n = cv2.GaussianBlur(self.rng.standard_normal((h, w)).astype(np.float32), (0, 0), max(0.5, self.ell))
            return n / (n.std() + 1e-9) * self.s
        nx, ny = draw(), draw()
        if self.fx is None or self.fx.shape != (h, w):
            self.fx, self.fy = nx, ny
        else:
            k = np.sqrt(1 - self.rho ** 2)
            self.fx = self.rho * self.fx + k * nx
            self.fy = self.rho * self.fy + k * ny
        return self.fx, self.fy


def apply_layer_fx(s_rgba, prof, ctx):
    """Effects on the resized sprite layer BEFORE compositing (motion blur, turbulence warp/blur)."""
    if "b" in prof and ctx["kind"] == "vehicle":
        s_rgba = motion_blur(s_rgba, *ctx["vel"])
    if "t1" in prof or "t2" in prof:
        st = ctx["stats"]
        h, w = s_rgba.shape[:2]
        pad = int(np.ceil(3 * st["turb_sigma"])) + 2
        s_p = cv2.copyMakeBorder(s_rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        fx, fy = ctx["turb"].step(*s_p.shape[:2])
        gy, gx = np.mgrid[0:s_p.shape[0], 0:s_p.shape[1]].astype(np.float32)
        s_p = _premul_filter(s_p, lambda x: cv2.remap(x, gx + fx, gy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        if "t2" in prof:
            sb = float(np.clip(0.5 * st["turb_sigma"], 0.3, 2.0)) * ctx["rng"].uniform(0.5, 1.5)
            s_p = _premul_filter(s_p, lambda x: cv2.GaussianBlur(x, (0, 0), sb))
        s_rgba = s_p[pad:pad + h, pad:pad + w]
    return s_rgba


def apply_colour_fx(rgb, a, base, bbox, prof, ctx):
    """Colour effects on the sprite rgb (float32, sprite-sized) given alpha a; returns rgb, darkness."""
    ring = ring_pixels(base, bbox)
    dark = float(np.clip(1.0 - np.median(ring @ L_W) / 110.0, 0.0, 1.0))
    if "a" in prof:
        rgb = hist_match(rgb, a, ring)
    if "b" in prof and ctx["kind"] == "vehicle":
        rgb = rgb * (1.0 - 0.6 * dark)
    if "c" in prof:
        st = ctx["stats"]
        L = lum(rgb)
        s = float(np.clip(0.8 * dark, 0.0, 0.7))
        rgb = rgb * (1 - s) + L[..., None] * s
        sd = np.sqrt(np.maximum(st["noise_a"] * L + st["noise_b"], 0.0))
        rgb = rgb + ctx["rng"].standard_normal(rgb.shape).astype(np.float32) * sd[..., None]
    return np.clip(rgb, 0, 255), dark
