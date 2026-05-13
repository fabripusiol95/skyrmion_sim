#pragma once
// NOTE: only include this header from .cu files — it pulls in Thrust/CUDA headers.

#include <thrust/device_vector.h>
#include "params.h"
#include "precision.h"

// ─────────────────────────────────────────────────────────────────────────────
//  SimState — all mutable simulation state that lives on the GPU.
//
//  All particle arrays have length N.
//  Pin arrays have length N_pins (may be empty).
//
//  After every completed integration step the following invariants hold:
//    x, y   → positions at time t (PBC-wrapped if p.pbc)
//    vx, vy → Thiele velocities at (x, y)
// ─────────────────────────────────────────────────────────────────────────────

struct SimState {
    int         N;
    int         N_pins;
    double      t;          // current simulation time
    long long   step;       // completed step count
    double      dt_current; // current step size (mirrors p.dt for fixed-step integrators)

    // ── Primary arrays ────────────────────────────────────────────────────
    thrust::device_vector<Real> x, y;    // positions   [N]
    thrust::device_vector<Real> vx, vy;  // velocities  [N]

    // ── Pin positions (fixed after initialisation) ────────────────────────
    thrust::device_vector<Real> px, py;  // [N_pins]

    // ── RK scratch ────────────────────────────────────────────────────────
    // x_tmp, y_tmp : intermediate stage positions         [N each]
    thrust::device_vector<Real> x_tmp, y_tmp;

    // ── Cell-linked list for pinning (built once by build_pin_cells) ──────
    thrust::device_vector<int> pin_list_cll;   // [N_pins] sorted pin indices
    thrust::device_vector<int> cell_start_cll; // [nx*ny]  start of cell c
    thrust::device_vector<int> cell_end_cll;   // [nx*ny]  end   of cell c
    int  pin_cell_nx       = 0;
    int  pin_cell_ny       = 0;
    Real pin_inv_cell_size = Real(0);
    bool has_pin_cells     = false;

    // ── Constructor ───────────────────────────────────────────────────────
    explicit SimState(const SimParams& p);
};
