import math
import numpy as np
from moviepy.editor import ImageClip, VideoClip, CompositeVideoClip
from PIL import Image

sigil_path = "scooty_sigil.png"
seal_path = "scooty_alchemy_seal.png"
crown_path = "scooty_soundwave_crown.png"

W, H = 1080, 1920
DUR = 6.0
FPS = 24
OUT_MP4 = "scooty_signature_loop_6s_1080x1920.mp4"

# Palette
OBSIDIAN = np.array([7, 6, 10], dtype=np.float32)
SAPPHIRE = np.array([8, 26, 75], dtype=np.float32)
PURPLE = np.array([59, 27, 120], dtype=np.float32)
CRIMSON = np.array([140, 17, 40], dtype=np.float32)

# Base gradient
y = np.linspace(0, 1, H, dtype=np.float32).reshape(H, 1, 1)
base = np.repeat(((1 - y) * OBSIDIAN + y * SAPPHIRE).reshape(H, 1, 3), W, axis=1)

yy, xx = np.mgrid[0:H, 0:W]
cx, cy = W / 2, H / 2
r = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2).astype(np.float32)
vig = np.clip(1 - 1.9 * r, 0.25, 1.0)[..., None]
scan = (0.988 + 0.012 * np.sin(2 * np.pi * (yy / 6.0))).astype(np.float32)[..., None]

# Periodic noise texture (tileable so looping is seamless)
np.random.seed(7)
noise_tex = np.random.rand(H, W).astype(np.float32)


def smoothstep(a, b, t):
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * (3 - 2 * u)


def make_bg(t):
    # periodic drift
    phase = 2 * math.pi * (t / DUR)
    dx = int(24 * math.sin(phase))
    dy = int(36 * math.cos(phase))
    n = np.roll(noise_tex, shift=(dy, dx), axis=(0, 1))

    frame = base + ((n - 0.5)[..., None] * (PURPLE * 0.22))
    frame = frame * scan
    frame = frame * vig

    # glitch pops at 2s and 4s (repeat each loop)
    for t0 in (2.0, 4.0):
        d = abs(t - t0)
        if d <= 0.12:
            g = 1 - d / 0.12
            px = int(10 * g)
            rch = np.roll(frame[:, :, 0], shift=px, axis=1)
            gch = np.roll(frame[:, :, 1], shift=-max(1, px // 2), axis=0)
            bch = np.roll(frame[:, :, 2], shift=-px, axis=1)
            frame = np.stack([rch, gch, bch], axis=2)
            frame = np.clip(frame + (0.05 * g * 255), 0, 255)

    return np.clip(frame, 0, 255).astype(np.uint8)


bg = VideoClip(make_bg, duration=DUR).set_fps(FPS)


def with_opacity(clip, op_func):
    c = clip.copy()
    if c.mask is None:
        c = c.add_mask()
    c.mask = c.mask.fl(lambda gf, tt: gf(tt) * op_func(tt))
    return c


def pulse(t, span=2.0):
    prog = (t % span) / span
    return 1.0 + 0.045 * math.sin(math.pi * prog)


def weights(t):
    w = 0.30
    if t < 2 - w:
        return (1, 0, 0)
    if t < 2 + w:
        a = smoothstep(2 - w, 2 + w, t)
        return (1 - a, a, 0)
    if t < 4 - w:
        return (0, 1, 0)
    if t < 4 + w:
        a = smoothstep(4 - w, 4 + w, t)
        return (0, 1 - a, a)
    return (0, 0, 1)


# Ensure alpha is respected by loading as RGBA
sig = ImageClip(sigil_path, transparent=True).set_duration(DUR).set_position("center")
seal = ImageClip(seal_path, transparent=True).set_duration(DUR).set_position("center")
crwn = ImageClip(crown_path, transparent=True).set_duration(DUR).set_position("center")

sig = sig.resize(lambda t: 0.62 * pulse(t))
seal = seal.resize(lambda t: 0.62 * pulse(t))
crwn = crwn.resize(lambda t: 0.62 * pulse(t))

sig = with_opacity(sig, lambda t: weights(t)[0])
seal = with_opacity(seal, lambda t: weights(t)[1])
crwn = with_opacity(crwn, lambda t: weights(t)[2])

# Ritual veil during morphs
veil_img = Image.new("RGB", (W, H), tuple(PURPLE.astype(int)))
veil = ImageClip(np.array(veil_img)).set_duration(DUR)
veil = with_opacity(veil, lambda t: 0.16 if (1.6 <= t <= 2.4 or 3.6 <= t <= 4.4) else 0.0)

# Crimson micro-hit near crown entrance
crim_img = Image.new("RGB", (W, H), tuple(CRIMSON.astype(int)))
crim = ImageClip(np.array(crim_img)).set_duration(DUR)
crim = with_opacity(crim, lambda t: 0.22 if (4.70 <= t <= 4.78) else 0.0)

final = CompositeVideoClip([bg, crim, veil, sig, seal, crwn], size=(W, H)).set_duration(DUR).set_fps(FPS)

final.write_videofile(
    OUT_MP4,
    fps=FPS,
    codec="libx264",
    audio=False,
    preset="ultrafast",
    bitrate="8000k",
)
