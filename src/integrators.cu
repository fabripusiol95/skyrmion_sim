#include "integrators.h"
#include "forces.h"
#include "precision.h"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cmath>
#include <stdexcept>
#include "nvtx.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Shared device kernels
// ─────────────────────────────────────────────────────────────────────────────

// out = base + a * k
__global__ static void axpy2d_k(
    Real* xout, Real* yout,
    const Real* xbase, const Real* ybase,
    const Real* kx, const Real* ky,
    Real a, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    xout[i] = xbase[i] + a * kx[i];
    yout[i] = ybase[i] + a * ky[i];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers shared by all integrators
// ─────────────────────────────────────────────────────────────────────────────

// Convenience: raw pointers from SimState
#define RAW(v) ((v).data().get())

// Build a PinCells view from SimState when the CLL has been constructed.
// Returns nullptr when no CLL is available (brute-force fallback).
static const PinCells* get_pin_cells(const SimState& s, PinCells& buf) {
    if (!s.has_pin_cells) return nullptr;
    buf = {RAW(s.pin_list_cll), RAW(s.cell_start_cll), RAW(s.cell_end_cll),
           s.pin_cell_nx, s.pin_cell_ny, s.pin_inv_cell_size};
    return &buf;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Euler
// ─────────────────────────────────────────────────────────────────────────────
void step_euler(SimState& s, const SimParams& p) {
    int blocks = (s.N + p.n_threads - 1) / p.n_threads;
    // x += dt * V(x)
    axpy2d_k<<<blocks, p.n_threads>>>(
        RAW(s.x), RAW(s.y), RAW(s.x), RAW(s.y),
        RAW(s.vx), RAW(s.vy), Real(p.dt), s.N);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RK2 (explicit midpoint / Heun)
//    k1 = dt * V(x)
//    k2 = dt * V(x + k1/2)
//    x += k2
// ─────────────────────────────────────────────────────────────────────────────
void step_rk2(SimState& s, const SimParams& p) {
    const Real* px = s.N_pins > 0 ? RAW(s.px) : nullptr;
    const Real* py = s.N_pins > 0 ? RAW(s.py) : nullptr;
    PinCells pc_buf; const PinCells* pc = get_pin_cells(s, pc_buf);
    int blocks = (s.N + p.n_threads - 1) / p.n_threads;

    // x_tmp = x + (1/2)*k1
    axpy2d_k<<<blocks, p.n_threads>>>(
        RAW(s.x_tmp), RAW(s.y_tmp),
        RAW(s.x), RAW(s.y),
        RAW(s.vx), RAW(s.vy), Real(0.5*p.dt), s.N);

    // V(x_tmp)
    compute_velocities(RAW(s.x_tmp), RAW(s.y_tmp), px, py,
             RAW(s.vx), RAW(s.vy), p, pc);

    // x += k2
    axpy2d_k<<<blocks, p.n_threads>>>(
        RAW(s.x), RAW(s.y), RAW(s.x), RAW(s.y),
        RAW(s.vx), RAW(s.vy), Real(p.dt), s.N);
}

#undef RAW
