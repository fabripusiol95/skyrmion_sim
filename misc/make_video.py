"""
Render trajectory frames to an animated GIF using Pillow (no ffmpeg needed).

Usage:
    python make_video.py <data_dir> <basename> [--fps 15] [--out video.gif] [--stride 1]

Example:
    python make_video.py results/movie movie --fps 15 --out movie.gif
"""

import os
import re
import argparse
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Render skyrmion trajectory to GIF')
parser.add_argument('data_dir')
parser.add_argument('basename')
parser.add_argument('--fps',    type=int, default=15)
parser.add_argument('--out',    default='video.gif')
parser.add_argument('--stride', type=int, default=1,
                    help='Use every Nth frame (1 = all frames)')
args = parser.parse_args()

dir_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(dir_root, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir

# ---------------------------------------------------------------------------
# Load simulation parameters from trajectory header
# ---------------------------------------------------------------------------
traj_path = os.path.join(data_dir, f'{args.basename}_traj.dat')
with open(traj_path) as f:
    f.readline()
    params_line = f.readline().strip().lstrip('#').strip()
params = dict(kv.split('=') for kv in params_line.split())
Lx, Ly = float(params['Lx']), float(params['Ly'])
N = int(params['N'])

# ---------------------------------------------------------------------------
# Parse all frames
# ---------------------------------------------------------------------------
steps = []
step_re = re.compile(r'#\s*step=(\d+)\s+t=([\deE+\-.]+)')

with open(traj_path) as f:
    current_meta = None
    current_rows = []
    for line in f:
        line = line.rstrip('\n')
        m = step_re.search(line)
        if m:
            if current_meta is not None and len(current_rows) == N:
                steps.append({**current_meta, 'pos': np.array(current_rows)})
            current_meta = {'step': int(m.group(1)), 't': float(m.group(2))}
            current_rows = []
        elif line.startswith('#') or line.strip() == '':
            continue
        else:
            current_rows.append([float(v) for v in line.split()])
    if current_meta is not None and len(current_rows) == N:
        steps.append({**current_meta, 'pos': np.array(current_rows)})

frames = steps[::args.stride]
print(f"Total frames: {len(steps)}  →  rendering {len(frames)} (stride={args.stride})")

# ---------------------------------------------------------------------------
# Load pins
# ---------------------------------------------------------------------------
pin = np.loadtxt(os.path.join(data_dir, f'{args.basename}_pins.dat'), comments='#')

# ---------------------------------------------------------------------------
# Build figure (static elements drawn once)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(pin[:, 1], pin[:, 2], 'o',
        color='grey', markersize=3, alpha=0.5, zorder=1)
for px, py in zip(pin[:, 1], pin[:, 2]):
    ax.add_patch(plt.Circle((px, py), 0.5,
                             color='grey', fill=True, alpha=0.1,
                             linewidth=0.5, zorder=1))
ax.add_patch(plt.Rectangle((0, 0), Lx, Ly,
                             linewidth=1.75, edgecolor='black',
                             facecolor='none', zorder=3))
ax.set_xlim(-0.5, Lx + 0.5)
ax.set_ylim(-0.5, Ly + 0.5)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_aspect(1)

pos0 = frames[0]['pos']
scat, = ax.plot(pos0[:, 1] % Lx, pos0[:, 2] % Ly, 'o',
                color='xkcd:dark red', markersize=10,
                markeredgecolor='gray', markeredgewidth=0.5, zorder=2)
title = ax.set_title(f"time = {frames[0]['t']:.3e}", fontsize=14)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Render frames → PIL images
# ---------------------------------------------------------------------------
pil_frames = []
duration_ms = int(1000 / args.fps)

for i, frame in enumerate(frames):
    pos = frame['pos']
    scat.set_xdata(pos[:, 1] % Lx)
    scat.set_ydata(pos[:, 2] % Ly)
    title.set_text(f"time = {frame['t']:.3e}")

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    pil_frames.append(Image.open(buf).copy())

    if (i + 1) % 50 == 0 or i == len(frames) - 1:
        print(f"  rendered {i+1}/{len(frames)}", flush=True)

plt.close(fig)

# ---------------------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------------------
out_path = args.out if os.path.isabs(args.out) else os.path.join(dir_root, args.out)
print(f"Saving → {out_path}")
pil_frames[0].save(
    out_path,
    save_all=True,
    append_images=pil_frames[1:],
    duration=duration_ms,
    loop=0,
)
print("Done.")
