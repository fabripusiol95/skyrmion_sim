#include "rng.h"
#include "precision.h"

#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <fstream>
#include <stdexcept>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Philox-based device functor: maps index i → uniform draw in [lo, hi).
//  Using (seed, i) as (key, counter) gives statistically independent streams
//  for different seeds without any host-side memory allocation.
// ─────────────────────────────────────────────────────────────────────────────
struct PhiloxUniform {
    uint64_t seed;
    double lo, hi;   // bounds kept as double for accurate range computation

    __device__ Real operator()(int i) const {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, static_cast<unsigned long long>(i), 0, &st);
        // curand_uniform_double returns (0,1]; shift to [0,1)
        double u = curand_uniform_double(&st);
        u = (u >= 1.0) ? 0.0 : u;
        return Real(lo + (hi - lo) * u);   // compute in double, store as Real
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  fill_uniform — helper that fills a device vector with the functor above.
//  Different seed values give independent draws (used for x vs y, etc.).
// ─────────────────────────────────────────────────────────────────────────────
static void fill_uniform(thrust::device_vector<Real>& v,
                         double lo, double hi, uint64_t seed) {
    thrust::counting_iterator<int> idx(0);
    thrust::transform(idx, idx + static_cast<int>(v.size()),
                      v.begin(), PhiloxUniform{seed, lo, hi});
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

void init_positions_triangular(SimState& s, const SimParams& p) {
    const int    N  = skN(p);
    const double Lx = skLx(p);
    const double Ly = skLy(p);
    const double dx = Lx / p.nx;
    const double dy = Ly / p.ny;

    // Compute in double, assign to device_vector<Real> (Thrust converts element-wise)
    thrust::host_vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        int col = i % p.nx;
        int row = i / p.nx;
        hx[i] = (col + 0.5 * (row % 2)) * dx;
        hy[i] = (row + 0.5) * dy;
    }
    s.x.assign(hx.begin(), hx.end());
    s.y.assign(hy.begin(), hy.end());
}

void init_positions_random(SimState& s, const SimParams& p) {
    const int N = skN(p);
    s.x.resize(N);
    s.y.resize(N);
    fill_uniform(s.x, 0.0, skLx(p), p.seed);
    fill_uniform(s.y, 0.0, skLy(p), p.seed + 1);
}

void init_positions_file(SimState& s, const SimParams& p) {
    std::ifstream f(p.init_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open init_file: " + p.init_file);

    const int N = skN(p);
    thrust::host_vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        if (!(f >> hx[i] >> hy[i]))
            throw std::runtime_error(
                "init_file ended prematurely — expected " +
                std::to_string(N) + " positions, got " + std::to_string(i));
    }
    s.x.assign(hx.begin(), hx.end());
    s.y.assign(hy.begin(), hy.end());
}

void init_pins_random(SimState& s, const SimParams& p) {
    // Generate candidates in batches on-GPU using Philox, reject on host.
    // Each batch uses a seed offset large enough not to collide with other
    // streams (seed+2 / seed+3 for first batch, +4/+5 for next, etc.).
    const double Lx = skLx(p);
    const double Ly = skLy(p);
    const int N_pins = s.N_pins;
    thrust::host_vector<double> hpx, hpy;
    hpx.reserve(N_pins);
    hpy.reserve(N_pins);

    const int batch = std::max(N_pins, 256);
    int batch_idx   = 0;   // counts how many batches have been drawn

    while (static_cast<int>(hpx.size()) < N_pins) {
        if (batch_idx > 1000)
            throw std::runtime_error(
                "init_pins_random: could not place all " +
                std::to_string(N_pins) +
                " pins after many batches — reduce n_p or r_p");

        thrust::device_vector<Real> d_cx(batch), d_cy(batch);
        fill_uniform(d_cx, 0.0, Lx, p.seed + 2 + 2 * batch_idx);
        fill_uniform(d_cy, 0.0, Ly, p.seed + 3 + 2 * batch_idx);
        ++batch_idx;

        // Copy to host as double for the rejection sampling arithmetic
        thrust::host_vector<double> cx(d_cx.begin(), d_cx.end());
        thrust::host_vector<double> cy(d_cy.begin(), d_cy.end());

        for (int k = 0; k < batch && static_cast<int>(hpx.size()) < N_pins; ++k) {
            bool ok = true;
            for (int j = 0; j < static_cast<int>(hpx.size()); ++j) {
                double ddx = cx[k] - hpx[j];
                double ddy = cy[k] - hpy[j];
                // Minimum-image convention.
                ddx -= Lx * std::round(ddx / Lx);
                ddy -= Ly * std::round(ddy / Ly);
                if (std::sqrt(ddx * ddx + ddy * ddy) < 2*p.r_p) { ok = false; break; }
            }
            if (ok) { hpx.push_back(cx[k]); hpy.push_back(cy[k]); }
        }
    }

    s.px.assign(hpx.begin(), hpx.end());
    s.py.assign(hpy.begin(), hpy.end());
}

void init_pins_file(SimState& s, const SimParams& p) {
    std::ifstream f(p.pin_file);
    if (!f.is_open())
        throw std::runtime_error("Cannot open pin_file: " + p.pin_file);

    const int N_pins = s.N_pins;
    thrust::host_vector<double> hpx(N_pins), hpy(N_pins);
    for (int i = 0; i < N_pins; ++i) {
        if (!(f >> hpx[i] >> hpy[i]))
            throw std::runtime_error(
                "pin_file ended prematurely — expected " +
                std::to_string(N_pins) + " positions, got " +
                std::to_string(i));
    }
    s.px.assign(hpx.begin(), hpx.end());
    s.py.assign(hpy.begin(), hpy.end());
}
