#include "io.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <numeric>
#include <filesystem>

namespace io {

// ─────────────────────────────────────────────────────────────────────────────
//  Module-level state
// ─────────────────────────────────────────────────────────────────────────────

static std::ofstream g_traj;
static std::ofstream g_thermo;

// Accumulate every thermo sample for end-of-run averaging
static std::vector<double> g_vx_hist, g_vy_hist;

// ─────────────────────────────────────────────────────────────────────────────
//  open_outputs
// ─────────────────────────────────────────────────────────────────────────────

void write_pins(int N_pins, const double* px_h, const double* py_h,
                const SimParams& p)
{
    std::filesystem::create_directories(p.output_dir);
    std::string path = p.output_dir + "/" + p.run_tag + "_pins.dat";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open pins file: " + path);

    f << "# Pin positions  N_pins=" << N_pins
      << "  r_p=" << p.r_p << "\n"
      << "# Columns: i  px  py\n";
    f << std::scientific << std::setprecision(10);
    for (int i = 0; i < N_pins; ++i)
        f << i << " " << px_h[i] << " " << py_h[i] << "\n";
}

void open_outputs(const SimParams& p) {
    std::filesystem::create_directories(p.output_dir);
    g_vx_hist.clear();
    g_vy_hist.clear();

    std::string traj_path = p.output_dir + "/" + p.run_tag + "_traj.dat";
    g_traj.open(traj_path);
    if (!g_traj.is_open())
        throw std::runtime_error("Cannot open trajectory file: " + traj_path);

    g_traj << "# Skyrmion trajectory\n"
           << "# N=" << skN(p)
           << "  Lx=" << skLx(p) << "  Ly=" << skLy(p)
           << "  alpha=" << p.alpha << "  G_z=" << p.G_z << "\n"
           << "# Columns: i  x  y";
    if (p.save_vel) g_traj << "  vx  vy";
    g_traj << "\n# (each block: step=<s> t=<t>)\n";

    std::string thermo_path = p.output_dir + "/" + p.run_tag + "_thermo.dat";
    g_thermo.open(thermo_path);
    if (!g_thermo.is_open())
        throw std::runtime_error("Cannot open thermo file: " + thermo_path);

    g_thermo << "# step  t  vx_mean  vy_mean  vmag_mean\n";
    g_thermo << std::scientific << std::setprecision(8);
}

// ─────────────────────────────────────────────────────────────────────────────
//  write_snapshot
// ─────────────────────────────────────────────────────────────────────────────

void write_snapshot(
    long long step, double t, int N,
    const double* x_h,  const double* y_h,
    const double* vx_h, const double* vy_h,
    const SimParams& /*p*/)
{
    g_traj << "# step=" << step << " t=" << std::scientific
           << std::setprecision(6) << t << "\n";
    for (int i = 0; i < N; ++i) {
        g_traj << i
               << " " << std::setprecision(10) << x_h[i]
               << " " << y_h[i];
        if (vx_h && vy_h)
            g_traj << " " << vx_h[i] << " " << vy_h[i];
        g_traj << "\n";
    }
    g_traj << "\n";
    g_traj.flush();
}

// ─────────────────────────────────────────────────────────────────────────────
//  write_thermo
// ─────────────────────────────────────────────────────────────────────────────

void write_thermo(
    long long step, double t,
    double vx_mean, double vy_mean,
    const SimParams& /*p*/)
{
    double vmag = std::sqrt(vx_mean*vx_mean + vy_mean*vy_mean);
    g_thermo << step
             << " " << t
             << " " << vx_mean
             << " " << vy_mean
             << " " << vmag
             << "\n";
    g_thermo.flush();

    g_vx_hist.push_back(vx_mean);
    g_vy_hist.push_back(vy_mean);
}

// ─────────────────────────────────────────────────────────────────────────────
//  close_outputs
//  Writes {run_tag}_summary.dat with key=value pairs including time-averaged
//  velocities over the second half of the run (steady-state estimate).
// ─────────────────────────────────────────────────────────────────────────────

void close_outputs(const SimParams& p) {
    if (g_traj.is_open())   g_traj.close();
    if (g_thermo.is_open()) g_thermo.close();

    // Time-average over the second half of collected thermo samples
    double vx_avg = 0.0, vy_avg = 0.0;
    if (!g_vx_hist.empty()) {
        std::size_t start = g_vx_hist.size() / 2;   // skip first half as transient
        std::size_t n     = g_vx_hist.size() - start;
        vx_avg = std::accumulate(g_vx_hist.begin() + start, g_vx_hist.end(), 0.0) / n;
        vy_avg = std::accumulate(g_vy_hist.begin() + start, g_vy_hist.end(), 0.0) / n;
    }
    double vmag_avg  = std::sqrt(vx_avg*vx_avg + vy_avg*vy_avg);
    double theta_H   = std::atan2(vy_avg, vx_avg) * 180.0 / 3.14159265358979323846;

    std::string sum_path = p.output_dir + "/" + p.run_tag + "_summary.dat";
    std::ofstream sf(sum_path);
    if (!sf.is_open()) return;   // non-fatal: summary is optional

    sf << std::scientific << std::setprecision(8);
    sf << "# Skyrmion simulation summary — " << p.run_tag << "\n";
    sf << "# Velocities averaged over the last 50% of thermo samples\n";
    sf << "nx            " << p.nx            << "\n";
    sf << "ny            " << p.ny            << "\n";
    sf << "N             " << skN(p)          << "\n";
    sf << "n_sk          " << p.n_sk          << "\n";
    sf << "Lx            " << skLx(p)         << "\n";
    sf << "Ly            " << skLy(p)         << "\n";
    sf << "alpha         " << p.alpha        << "\n";
    sf << "G_z           " << p.G_z          << "\n";
    sf << "K1            " << p.K1           << "\n";
    sf << "lambda        " << p.lambda       << "\n";
    sf << "n_p           " << p.n_p          << "\n";
    sf << "N_pins        " << skN_pins(p)   << "\n";
    sf << "f_p           " << p.f_p          << "\n";
    sf << "r_p           " << p.r_p          << "\n";
    sf << "F_D           " << p.F_D          << "\n";
    sf << "drive_angle   " << p.drive_angle  << "\n";
    sf << "t_max         " << p.t_max        << "\n";
    sf << "n_samples     " << g_vx_hist.size()<< "\n";
    sf << "vx_avg        " << vx_avg         << "\n";
    sf << "vy_avg        " << vy_avg         << "\n";
    sf << "vmag_avg      " << vmag_avg       << "\n";
    sf << "theta_H_deg   " << theta_H        << "\n";

    g_vx_hist.clear();
    g_vy_hist.clear();
}

} // namespace io
