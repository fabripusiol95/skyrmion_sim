#pragma once
// Pure C++ header — no CUDA/Thrust types.
#include "params.h"
#include <cstdint>

namespace io {

// Write pin positions to {output_dir}/{run_tag}_pins.dat.
// Creates output_dir if needed.  Safe to call before open_outputs.
void write_pins(int N_pins, const double* px_h, const double* py_h,
                const SimParams& p);

// Open output files.  Creates output_dir if it does not exist.
void open_outputs(const SimParams& p);

// Write a position (and optionally velocity) snapshot from host arrays.
// vx_h / vy_h may be nullptr when p.save_vel is false.
void write_snapshot(
    long long step, double t, int N,
    const double* x_h,  const double* y_h,
    const double* vx_h, const double* vy_h,
    const SimParams& p);

// Write one line to the thermodynamics file.
void write_thermo(
    long long step, double t,
    double vx_mean, double vy_mean,
    const SimParams& p);

// Flush and close all output streams.
// Also writes {output_dir}/{run_tag}_summary.dat: one key=value line per
// physical quantity, with time-averaged velocities over the second half of
// the run.  Useful for collecting V-F curve data across many runs.
void close_outputs(const SimParams& p);

} // namespace io
