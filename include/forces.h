#pragma once
#include "params.h"
#include "precision.h"

// ─────────────────────────────────────────────────────────────────────────────
//  PinCells — cell-linked list metadata for the pinning force kernel.
//
//  Built once on the host (via build_pin_cells) after pins are initialised.
//  All pointer members are device pointers; nx/ny/inv_cell_size describe the
//  uniform grid that tiles the simulation box with cells of side r_p.
// ─────────────────────────────────────────────────────────────────────────────
struct PinCells {
    const int*    pin_list;    // [N_pins] pin indices sorted by cell
    const int*    cell_start;  // [nx*ny]  first entry in pin_list for cell c
    const int*    cell_end;    // [nx*ny]  one-past-last entry for cell c
    int    nx, ny;
    Real   inv_cell_size;      // 1 / cell_size  (cell_size == r_p)
};

// Forward declaration — full type in state.h (CUDA/.cu only)
struct SimState;

// Build the cell-linked list from the pin arrays already stored in s.
// Must be called after pin positions are loaded; call only from .cu files.
void build_pin_cells(SimState& s, const SimParams& p);

// ─────────────────────────────────────────────────────────────────────────────
//  compute_velocities
//
//  Computes the Thiele velocity of each skyrmion and writes it to vx_d / vy_d.
//  All arrays are device pointers of length p.N (or p.N_pins for px/py).
//  px_d / py_d may be nullptr when p.N_pins == 0.
//  pc may be nullptr to use the O(N·M) brute-force pinning kernel.
// ─────────────────────────────────────────────────────────────────────────────
void compute_velocities(
    const Real* x_d,  const Real* y_d,
    const Real* px_d, const Real* py_d,
    Real* vx_d, Real* vy_d,
    const SimParams& p,
    const PinCells* pc = nullptr);
