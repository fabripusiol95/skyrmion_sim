#pragma once
// NOTE: only include from .cu files (depends on state.h / CUDA).

#include "state.h"
#include "params.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Integration step functions
//  Each stepper advances positions by dt using s.vx/vy computed at the current
//  step.  The caller is responsible for calling compute_velocities after each
//  step to refresh s.vx/vy at the new positions.
// ─────────────────────────────────────────────────────────────────────────────
void   step_euler(SimState& s, const SimParams& p);
void   step_rk2  (SimState& s, const SimParams& p);