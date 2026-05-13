import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

dir_root = r'/home/fabri/Dropbox/IB_PhD/PhD_vault/skyrmion_sim'
dir_data = sys.argv[1]

traj_path = os.path.join(dir_data, r'euler_test_traj.dat')
with open(traj_path) as f:
    f.readline()  # skip first comment line
    params_line = f.readline().strip().lstrip('#').strip()
params = dict(kv.split('=') for kv in params_line.split())
Lx, Ly = float(params['Lx']), float(params['Ly'])
N = int(params['N'])

# Parse all steps from trajectory file
steps = []      # list of dicts: {step, t, pos}
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

n_steps = len(steps)

pin = np.loadtxt(os.path.join(dir_data, r'euler_test_pins.dat'),
                 comments='#',
                 skiprows=0)

fig, ax = plt.subplots(figsize=(6, 6))
fig.subplots_adjust(bottom=0.12)

ax.plot(pin[:, 1], pin[:, 2], 'o', 
        color='grey', 
        markersize=7, 
        label='pins', 
        alpha=0.5)
for px, py in zip(pin[:, 1], pin[:, 2]):
    ax.add_patch(plt.Circle((px, py), 0.5, color='grey', fill=True, alpha=0.1, linewidth=0.5))
ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, linewidth=1.5, edgecolor='black', facecolor='none'))
ax.set_xlim(-0.5, Lx + 0.5)
ax.set_ylim(-0.5, Ly + 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect(1)

pos0 = steps[0]['pos']
scat = ax.plot(pos0[:, 1] % Lx, pos0[:, 2] % Ly, 'o', 
               color='xkcd:dark red', 
               markersize=15,
               markeredgecolor='gray',
               markeredgewidth=1)[0]
title = ax.set_title(f"step={steps[0]['step']}  t={steps[0]['t']:.3e}")

ax_slider = fig.add_axes([0.15, 0.03, 0.7, 0.04])
slider = Slider(ax_slider, 'Step', 0, n_steps - 1, valinit=0, valstep=1)

def update(val):
    idx = int(val)
    pos = steps[idx]['pos']
    scat.set_xdata(pos[:, 1] % Lx)
    scat.set_ydata(pos[:, 2] % Ly)
    title.set_text(f"step={steps[idx]['step']}  t={steps[idx]['t']:.3e}")
    fig.canvas.draw()

slider.on_changed(update)

plt.show()
