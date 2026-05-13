#include "simulation.h"
#include "state.h"
#include "forces.h"
#include "integrators.h"
#include "io.h"
#include "rng.h"
#include "precision.h"
#include "nvtx.h"

#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  SimState constructor — allocates device memory based on integrator choice
// ─────────────────────────────────────────────────────────────────────────────

SimState::SimState(const SimParams& p)
    : N(skN(p)), N_pins(skN_pins(p)),
      t(0.0), step(0),
      dt_current(p.dt)
{
    // Primary arrays
    x.resize(N);  y.resize(N);
    vx.resize(N, Real(0)); vy.resize(N, Real(0));

    // Pin arrays (only when needed)
    if (N_pins > 0) {
        px.resize(N_pins);
        py.resize(N_pins);
    }

    // RK scratch — allocate based on integrator
    if (p.integrator == Integrator::RK2) {
        x_tmp.resize(N); y_tmp.resize(N);
    }

}

// Positions are kept in unwrapped (unfolded) coordinates throughout.
// PBC enters only through the minimum-image convention inside ()
// (the remainder() calls in min_image)

// ─────────────────────────────────────────────────────────────────────────────
//  Output helpers (handle D→H copy, then call io::)
// ─────────────────────────────────────────────────────────────────────────────

static void do_snapshot(const SimState& s, const SimParams& p) {
    // Copy device Real arrays to host double for I/O (Thrust converts element-wise)
    thrust::host_vector<double> hx(s.x.begin(), s.x.end());
    thrust::host_vector<double> hy(s.y.begin(), s.y.end());
    const double* hvx_ptr = nullptr;
    const double* hvy_ptr = nullptr;
    thrust::host_vector<double> hvx, hvy;
    if (p.save_vel) {
        hvx.assign(s.vx.begin(), s.vx.end());
        hvy.assign(s.vy.begin(), s.vy.end());
        hvx_ptr = hvx.data();
        hvy_ptr = hvy.data();
    }
    io::write_snapshot(s.step, s.t, s.N,
                       hx.data(), hy.data(),
                       hvx_ptr, hvy_ptr, p);
}

static void do_thermo(const SimState& s, const SimParams& p) {
    double vx_mean = double(thrust::reduce(s.vx.begin(), s.vx.end(), Real(0))) / s.N;
    double vy_mean = double(thrust::reduce(s.vy.begin(), s.vy.end(), Real(0))) / s.N;
    io::write_thermo(s.step, s.t, vx_mean, vy_mean, p);
}

// Forces + Thiele velocities at the current positions.
static void update_vel(SimState& s, const SimParams& p) {
    const Real* px = s.N_pins > 0 ? s.px.data().get() : nullptr;
    const Real* py = s.N_pins > 0 ? s.py.data().get() : nullptr;
    PinCells pc_buf;
    const PinCells* pc = nullptr;
    if (s.has_pin_cells) {
        pc_buf = {s.pin_list_cll.data().get(),
                  s.cell_start_cll.data().get(),
                  s.cell_end_cll.data().get(),
                  s.pin_cell_nx, s.pin_cell_ny, s.pin_inv_cell_size};  // inv_cell_size is Real
        pc = &pc_buf;
    }
    compute_velocities(s.x.data().get(), s.y.data().get(), px, py,
                   s.vx.data().get(), s.vy.data().get(), p, pc);
}

// ─────────────────────────────────────────────────────────────────────────────
//  namespace simulation::run
// ─────────────────────────────────────────────────────────────────────────────

namespace simulation {

void run(const SimParams& p) {

    // ── 1. Initialise state ──────────────────────────────────────────────────
    SimState s(p);

    switch (p.init_layout) {
        case InitLayout::TRIANGULAR: init_positions_triangular(s, p); break;
        case InitLayout::RANDOM:     init_positions_random(s, p);     break;
        case InitLayout::FILE:       init_positions_file(s, p);       break;
    }

    if (s.N_pins > 0) {
        if (p.pin_file.empty()) init_pins_random(s, p);
        else                    init_pins_file(s, p);

        thrust::host_vector<double> hpx(s.px.begin(), s.px.end());
        thrust::host_vector<double> hpy(s.py.begin(), s.py.end());
        io::write_pins(s.N_pins, hpx.data(), hpy.data(), p);

        if (p.use_pin_cells)
            build_pin_cells(s, p);
    }

    // ── 2. Compute initial forces & velocities ───────────────────────────────
    cudaDeviceSynchronize();
    update_vel(s, p);
    // {
    //     NvtxRange r_vel("update_vel");
    //     update_vel(s, p);
    // }
    cudaDeviceSynchronize();

    // ── 3. Open output files & write initial snapshot ────────────────────────
    io::open_outputs(p);
    do_snapshot(s, p);
    do_thermo(s, p);

    if (p.verbose) {
        std::cout << "[sim] Starting. N=" << skN(p)
                  << "  N_pins=" << s.N_pins
                  << "  t_max=" << p.t_max
                  << "\n";
    }

    auto check_output = [&]() {
        if (p.thermo_every > 0 && s.step % p.thermo_every == 0)
            do_thermo(s, p);
        if (p.save_every > 0 && s.step % p.save_every == 0)
            do_snapshot(s, p);
    };

    // ── 4. Main time loop ────────────────────────────────────────────────────
    //  Fixed-step (Euler/RK2): loop for n_steps = t_max/dt iterations.
    long long n_steps = static_cast<long long>(p.t_max / p.dt);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    for (long long i = 0; i < n_steps; ++i) {

        switch (p.integrator) {
                case Integrator::EULER: step_euler(s, p); break;
                case Integrator::RK2:   step_rk2  (s, p); break;
                default: break;
            }

        update_vel(s, p);
        // {
        //     NvtxRange r_vel("update_vel");
        //     update_vel(s, p);
        // }

        s.t    += p.dt;
        s.step += 1;
        
        check_output();

    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    if (p.verbose) {
        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev_start, ev_stop);
        std::cout << "[timer] main loop: total=" << total_ms << " ms"
                  << "  steps=" << n_steps
                  << "  avg=" << total_ms / n_steps << " ms/step\n";
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // ── 5. Final output & cleanup ────────────────────────────────────────────
    // Always write one last thermo + snapshot at the end (if not already done)
    if (p.thermo_every <= 0 || s.step % p.thermo_every != 0) do_thermo(s, p);
    if (p.save_every   <= 0 || s.step % p.save_every   != 0) do_snapshot(s, p);

    io::close_outputs(p);

    if (p.verbose)
        std::cout << "[sim] Done. t=" << s.t << "  steps=" << s.step << "\n";
}

} // namespace simulation
