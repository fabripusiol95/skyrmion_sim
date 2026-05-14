# Skyrmions on Random Pinning

GPU-accelerated 2D skyrmion dynamics simulator based on the overdamped Thiele equation,
implementing the model from [Reichhardt & Reichhardt, PRB 99, 104418 (2019)](https://link.aps.org/doi/10.1103/PhysRevB.99.104418).

## Physics

Each skyrmion obeys the Thiele equation:

```
α·v + G × v = F_ss + F_pin + F_D
```

where `α` is the damping coefficient, `G = G_z ẑ` is the gyrovector, and the three forces are:

- **F_ss** — skyrmion–skyrmion repulsion via modified Bessel K₁
- **F_pin** — parabolic potential wells placed at random (or from file)
- **F_D** — uniform applied drive with tunable magnitude and direction


---

## Building

Option 1. To build manually with a custom CUDA architecture:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc)
```

Option 2. Run the script `build.sh` to compile the program. For running on an HPC cluster, modify the loaded modules and CUDA architecture if necessary:

```bash
module load gcc/8.2.0
module load cuda/11.4.0
module load cmake-3.19.2-gcc-8.2.0
```

```bash
bash build.sh # loads modules, cleans, configures, and compiles in Release mode
```

CMake options:

| Flag | Default | Description |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `70;75;80;86` | GPU SM targets |
| `SK_FLOAT` | `ON` | Single-precision GPU arithmetic (faster); `OFF` for double |

The binary is written to `build/skyrmion_sim`.



---

## Usage

```bash
./build/skyrmion_sim <config.conf>
```

The simulator reads all parameters from the config file and writes four output files to `output_dir/`.

### Example

```bash
./build/skyrmion_sim inputs/example.conf
```

---

## Configuration

Config files use an INI-style format. Lines beginning with `#` or `;` are comments.
Keys are case-insensitive; only keys you want to override need to appear
(all parameters have sensible defaults). See [inputs/example.conf](inputs/example.conf) for the full reference.

### Key sections

**`[system]`** — lattice geometry

| Key | Description |
|---|---|
| `nx`, `ny` | Grid of skyrmions; total `N = nx × ny` |
| `n_sk` | Skyrmion density → lattice constant `a = sqrt(sqrt(3)/(2·n_sk))` |
| `pbc` | Periodic boundary conditions (`true`/`false`) |

**`[thiele]`** — equation of motion

| Key | Description |
|---|---|
| `alpha` | Damping coefficient αD |
| `g_z` | Gyrovector z-component; Hall angle `k = g_z/alpha` |

**`[skyrmion_skyrmion]`** — inter-skyrmion repulsion

| Key | Description |
|---|---|
| `K1` | Interaction prefactor |
| `lambda` | Interaction range (in units of `a₀`) |

**`[pinning]`** — disorder

| Key | Description |
|---|---|
| `n_p` | Pin density (pins per unit area); `N_pins = round(n_p · Lx · Ly)` |
| `f_p` | Maximum pin force magnitude |
| `r_p` | Pin capture radius (in units of `a₀`) |
| `pin_file` | *(optional)* Path to a file with explicit pin positions |

**`[drive]`** — applied force

| Key | Description |
|---|---|
| `F_D` | Drive magnitude |
| `drive_angle` | Drive direction in degrees (0 = +x) |

**`[integration]`**

| Key | Description |
|---|---|
| `integrator` | `euler` or `rk2` |
| `dt` | Time step |
| `t_max` | Total simulation time |

**`[init]`** — initial positions

| Key | Description |
|---|---|
| `init_layout` | `triangular` (perfect lattice), `random`, or `file` |
| `seed` | RNG seed |
| `init_file` | Path to positions file when `init_layout = file` |

**`[output]`**

| Key | Description |
|---|---|
| `output_dir` | Directory for output files (created if absent) |
| `run_tag` | Prefix for all output filenames |
| `save_every` | Steps between trajectory snapshots |
| `thermo_every` | Steps between scalar diagnostics |
| `save_vel` | Also write velocities in the trajectory file (`true`/`false`) |

**`[misc]`**

| Key | Description |
|---|---|
| `n_threads` | CUDA threads per block (default 256) |
| `verbose` | Print timing and progress to stdout |
| `use_pin_cells` | Use cell-linked list for O(N) pin search (recommended for large N) |

---

## Output files

All files are written to `output_dir/` with the prefix `run_tag`.

### `{run_tag}_traj.dat` — position snapshots

```
# Skyrmion trajectory
# N=256  Lx=37.2  Ly=32.2  alpha=1.0  G_z=0.79
# Columns: i  x  y
# (each block: step=<s> t=<t>)
# step=0 t=0.000000e+00
0  1.234  5.678
1  2.345  6.789
...
```

A block is written every `save_every` steps. If `save_vel = true`, columns `vx vy` are appended.
Positions are in unfolded (unwrapped) coordinates; PBC applies via minimum-image in force calculations.

### `{run_tag}_thermo.dat` — scalar time series

```
# step  t  vx_mean  vy_mean  vmag_mean
0  0.00e+00  2.47e-03  -1.91e-03  3.12e-03
100  1.00e+01  ...
```

One row every `thermo_every` steps: mean velocity components and magnitude averaged over all skyrmions.

### `{run_tag}_summary.dat` — end-of-run summary

Key–value file written once at the end. Contains all physical parameters plus time-averaged velocities
computed over the **last 50%** of thermo samples (steady-state estimate):

```
vx_avg        2.47e-03
vy_avg        -1.91e-03
vmag_avg      3.12e-03
theta_H_deg   -37.7
```

`theta_H_deg` is the Hall angle of the mean velocity vector.

### `{run_tag}_pins.dat` — pin positions

```
# Pin positions  N_pins=512  r_p=0.5
# Columns: i  px  py
0  384.52  512.30
...
```

Written once at the start of the run. Useful for reproducing or visualising the disorder configuration.

---

## Parameter sweeps

The [scripts/run_sweep.sh](scripts/run_sweep.sh) script loops over a range of `F_D` values,
patching the base config via `sed` for each point:

```bash
bash scripts/run_sweep.sh
```

Results land in `results/depinning/`.

---

## Analysis scripts

All scripts live in [scripts/](scripts/) and expect data in the standard output format above.

| Script | Purpose |
|---|---|
| `plot_step.py` | Slider: skyrmion trajectories over steps. |
| `plot_depinning.py` | Plot <vx> vs FD (results from `run_sweep.sh`) |


---

## Code structure

```
skyrmion_sim/
├── src/
│   ├── main.cu          entry point — parse config, call simulation::run()
│   ├── simulation.cu    main loop, state init, output scheduling
│   ├── forces.cu        F_ss, F_pin, F_D kernels; Thiele velocity solver
│   ├── integrators.cu   Euler and RK2 step kernels
│   ├── io.cpp           ASCII output (traj, thermo, summary, pins)
│   ├── params.cpp       INI config parser
│   └── rng.cu           cuRAND wrapper
├── include/             matching headers + state.h, precision.h
├── inputs/              example .conf files
├── scripts/             Python plots and sweep scripts
└── build.sh             one-shot build helper
```