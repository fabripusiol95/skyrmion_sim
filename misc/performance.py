import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

path_font = '/home/fabri/Dropbox/IB_PhD/DWs/fonts/inter'
fonts = [fm.fontManager.addfont(os.path.join(path_font, file)) 
         for file in os.listdir(path_font) if file.endswith('.ttf')]

plt.style.use('scripts/plots.mplstyle')


base = os.path.join(os.path.dirname(__file__), "..", "results", "performance")

data_t = np.loadtxt(os.path.join(base, "bench_scaling_true.txt"), skiprows=1)
data_f = np.loadtxt(os.path.join(base, "bench_scaling_false.txt"), skiprows=1)

nx2_t = data_t[:, 0] ** 2
nx2_f = data_f[:, 0] ** 2
ms_t  = data_t[:, 2]
ms_f  = data_f[:, 2]

ms = 2.5
lw = 1.0
fig, ax = plt.subplots(figsize=(8/2.54, 7/2.54), layout="constrained")
ax.plot(nx2_t, ms_t, marker="o", ms=ms, lw=lw, 
        label="Cell linked list")
ax.plot(nx2_f, ms_f, marker="s", ms=ms, lw=lw, 
        label="Naive Algorithm")
ax.loglog()
ax.set_xlabel(r"# of Skyrmions $N$")
ax.set_ylabel("ms / step")
ax.set_title("Performance scaling")
ax.grid(True, which="major", ls="--", alpha=0.5, lw=0.5)
ax.legend()
plt.tight_layout()
plt.show()
