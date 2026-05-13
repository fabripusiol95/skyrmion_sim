#pragma once
// NOTE: only include from .cu files.
#include "state.h"
#include "params.h"

// Place skyrmions on a triangular lattice with the best divisor factorization
// N = nx*ny matching the target aspect ratio ny/nx = 2*Ly/(sqrt(3)*Lx).
void init_positions_triangular(SimState& s, const SimParams& p);

// Place skyrmions at uniform-random positions in [0,Lx) × [0,Ly).
void init_positions_random(SimState& s, const SimParams& p);

// Load positions from a two-column text file (x y per line).
void init_positions_file(SimState& s, const SimParams& p);

// Fill s.px, s.py with uniform random pin positions in [0,Lx) × [0,Ly).
void init_pins_random(SimState& s, const SimParams& p);

// Load pin positions from a two-column text file.
void init_pins_file(SimState& s, const SimParams& p);
