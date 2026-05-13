#pragma once
#include <string>
#include <cstdint>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  SimParams — all physical and numerical parameters for a skyrmion run.
//
//  Loaded from an INI-style text file (see inputs/example.conf).
//  All fields have a documented default so unset keys are non-fatal.
// ─────────────────────────────────────────────────────────────────────────────

enum class Integrator { EULER, RK2 };
enum class InitLayout  { TRIANGULAR, RANDOM, FILE };

struct SimParams {
    // ── System ──────────────────────────────────────────────────────────────
    // Layout: perfect triangular lattice with lattice constant a = sqrt(sqrt(3)/(2*n_sk))
    //   Lx = a * nx,  Ly = (sqrt(3)/2) * a * ny,  N = nx * ny  (all derived)
    int         nx          = 10;       // skyrmions along x
    int         ny          = 10;       // skyrmions along y
    double      n_sk        = 1.0;      // skyrmion density (skyrmions per unit area)
    bool        pbc         = true;     // periodic boundary conditions

    // ── Thiele equation coefficients ────────────────────────────────────────
    // Equation of motion:  alpha*v + G x v = F_total
    //   alpha  = damping (alpha_D in Reichhardt)
    //   G      = gyro coupling (G_z, sign depends on topological charge)
    // Derived Hall angle: k = G/alpha  =>  v solved analytically
    double      alpha       = 1.0;      // dimensionless damping
    double      G_z         = 1.0;      // gyrovector z-component (G/4pi conventionally)

    // ── Skyrmion–skyrmion interaction (modified Bessel K1) ──────────────────
    //   F_ss = K1 * sum_{j!=i} K1(r_ij / lambda) * r_hat_ij
    double      K1          = 1.0;      // overall ss interaction strength
    double      lambda      = 1.0;     // interaction range (in a_0 units)

    // ── Pinning (parabolic traps) ────────────────────────────────────────────
    double      n_p         = 0.0;      // pin density (pins per unit area)
    double      f_p         = 0.0;      // max pinning force magnitude
    double      r_p         = 0.5;      // pin radius (a_0 units)
    std::string pin_file    = "";       // path to pin positions file (optional)

    // ── Drive ────────────────────────────────────────────────────────────────
    double      F_D         = 0.0;      // drive magnitude
    double      drive_angle = 0.0;      // drive angle in degrees (0 = +x)

    // ── Integration ──────────────────────────────────────────────────────────
    Integrator  integrator  = Integrator::RK2;
    double      dt          = 0.01;     // time step (Euler/RK2)
    double      t_max       = 100.0;   // total simulation time


    // ── Initial configuration ────────────────────────────────────────────────
    InitLayout  init_layout = InitLayout::TRIANGULAR;
    std::string init_file   = "";       // path when init_layout = FILE
    uint64_t    seed        = 42;

    // ── Output ───────────────────────────────────────────────────────────────
    std::string output_dir  = ".";
    std::string run_tag     = "run";    // prefix for output files
    int         save_every  = 100;      // steps between position snapshots
    bool        save_vel    = false;    // also dump velocities
    bool        save_forces = false;    // also dump forces
    int         thermo_every= 10;       // steps between scalar diagnostics

    // ── Misc ─────────────────────────────────────────────────────────────────
    int         n_threads   = 256;      // CUDA threads per block
    bool        verbose     = false;
    bool        use_pin_cells = false;  // use cell-linked list for pinning (O(N) vs O(N·M))
};

// ─────────────────────────────────────────────────────────────────────────────
//  Parse a .conf file and fill a SimParams struct.
//  Throws std::runtime_error on any unrecognised key or malformed value.
// ─────────────────────────────────────────────────────────────────────────────
SimParams parse_params(const std::string& filename);

// Derived convenience accessors (computed, not stored)
inline double hall_angle(const SimParams& p) { return p.G_z / p.alpha; }  // k = G/alpha
inline double inv_denom(const SimParams& p) { return 1.0 / (1.0 + hall_angle(p) * hall_angle(p)); } // 1/(1+k²)

inline int    skN    (const SimParams& p) { return p.nx * p.ny; } // total number of skyrmions
inline double skA    (const SimParams& p) { return std::sqrt(std::sqrt(3.0) / (2.0 * p.n_sk)); }  // lattice constant a_0 for given n_sk 
inline double skLx   (const SimParams& p) { return skA(p) * p.nx; }
inline double skLy   (const SimParams& p) { return (std::sqrt(3.0) / 2.0) * skA(p) * p.ny; }
inline int    skN_pins(const SimParams& p) {
    return static_cast<int>(std::round(p.n_p * skLx(p) * skLy(p)));
}
inline double FD_x(const SimParams& p) {
    constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
    return p.F_D * std::cos(p.drive_angle * DEG2RAD);
}
inline double FD_y(const SimParams& p) {
    constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
    return p.F_D * std::sin(p.drive_angle * DEG2RAD);
}
inline double v0_x(const SimParams& p) {
    return inv_denom(p) * (FD_x(p) + hall_angle(p) * FD_y(p));
} 
inline double v0_y(const SimParams& p) {
    return inv_denom(p) * (FD_y(p) - hall_angle(p) * FD_x(p));
}