#include "forces.h"
#include "state.h"
#include "precision.h"
#include "nvtx.h"

#include <cmath>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// ═════════════════════════════════════════════════════════════════════════════
//  Device helper: modified Bessel functions I1 and K1
//
//  Polynomial approximations from Abramowitz & Stegun (§9.8) as tabulated
//  in Numerical Recipes in C (2nd ed.), routines bessi1 / bessk1.
//  Absolute error < 1.6×10⁻⁷ across (0, ∞).
//  Written with Real so that in float mode all arithmetic stays in float
//  (CUDA overloads exp/sqrt/fabs/log on argument type).
// ═════════════════════════════════════════════════════════════════════════════

// I₁(x)  — modified Bessel function, first kind, order 1  (x ≥ 0)
__device__ static Real bessi1_dev(Real x) {
    Real ax = fabs(x);
    Real ans;
    if (ax < Real(3.75)) {
        Real y = x / Real(3.75);
        y *= y;
        ans = ax * (Real(0.5)
            + y * (Real(0.87890594)
            + y * (Real(0.51498869)
            + y * (Real(0.15084934)
            + y * (Real(0.2658733e-1)
            + y * (Real(0.301532e-2)
            + y *  Real(0.32411e-3)))))));
    } else {
        Real y = Real(3.75) / ax;
        ans = (Real(0.39894228)
            + y * (Real(-0.3988024e-1)
            + y * (Real(-0.362018e-2)
            + y * (Real( 0.163801e-2)
            + y * (Real(-0.1031555e-1)
            + y * (Real( 0.2282967e-1)
            + y * (Real(-0.2895312e-1)
            + y * (Real( 0.1787654e-1)
            + y *  Real(-0.420059e-2)))))))));
        ans *= exp(ax) / sqrt(ax);
    }
    return (x < Real(0)) ? -ans : ans;
}

// K₁(x)  — modified Bessel function, second kind, order 1  (x > 0)
__device__ static Real bessk1_dev(Real x) {
    Real ans;
    if (x <= Real(2)) {
        Real y = x * x * Real(0.25);
        ans = (log(x * Real(0.5)) * bessi1_dev(x))
            + (Real(1) / x) * (Real(1)
            + y * (Real( 0.15443144)
            + y * (Real(-0.67278579)
            + y * (Real(-0.18156897)
            + y * (Real(-0.1919402e-1)
            + y * (Real(-0.110404e-2)
            + y *  Real(-0.4686e-4)))))));
    } else {
        Real y = Real(2) / x;
        ans = (exp(-x) / sqrt(x)) * (Real(1.25331414)
            + y * (Real( 0.23498619)
            + y * (Real(-0.3655620e-1)
            + y * (Real( 0.1504268e-1)
            + y * (Real(-0.780353e-2)
            + y * (Real( 0.325614e-2)
            + y *  Real(-0.68245e-3)))))));
    }
    return ans;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Device helper: minimum-image displacement (modifies dx, dy in place)
// ─────────────────────────────────────────────────────────────────────────────
__device__ static void min_image(Real& dx, Real& dy,
                                  Real Lx, Real Ly, bool pbc) {
    if (pbc) {
        dx = remainder(dx, Lx);
        dy = remainder(dy, Ly);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  F_ss — skyrmion–skyrmion repulsion
//
//  Force on particle i from all j ≠ i:
//    F_ss^i = K1_coeff · Σ_{j≠i}  K₁(r_ij / λ)  r̂_{ij}
//
//  Tiled shared-memory kernel: each block of T threads iterates over N/T
//  tiles, loading a tile of T (xj, yj) pairs into shared memory.  Every
//  thread in the block then iterates over the tile in the inner loop.
//  This halves global memory traffic compared to a naive kernel.
//  The drive force F_D is added in the same pass by initialising vx/vy to 
//  the drive velocity vd0 before adding the ss contribution.
//
//  Complexity: O(N²) per force evaluation.
// ═════════════════════════════════════════════════════════════════════════════
__global__ static void force_ss_k(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    Real* vx, Real* vy,
    Real K1_coeff, Real inv_lambda,
    Real Lx, Real Ly, Real hall_k, Real factor,
    Real init_vx, Real init_vy,
    bool pbc, int N)
{
    // Shared memory: interleaved x/y so both fit in one allocation.
    // Layout: [0 .. T-1] = sx,  [T .. 2T-1] = sy
    extern __shared__ Real sh[];
    Real* sx = sh;
    Real* sy = sh + blockDim.x;

    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    Real xi  = (i < N) ? x[i] : Real(0);
    Real yi  = (i < N) ? y[i] : Real(0);
    Real fxi = Real(0), fyi = Real(0);

    int n_tiles = (N + (int)blockDim.x - 1) / (int)blockDim.x;

    for (int tile = 0; tile < n_tiles; ++tile) {
        // Load tile of j particles (coalesced global reads)
        int j = tile * blockDim.x + threadIdx.x;
        sx[threadIdx.x] = (j < N) ? x[j] : Real(0);
        sy[threadIdx.x] = (j < N) ? y[j] : Real(0);
        __syncthreads();

        if (i < N) {
            int jend = min((int)blockDim.x, N - tile * (int)blockDim.x);
            for (int jt = 0; jt < jend; ++jt) {
                int jj = tile * blockDim.x + jt;
                if (jj == i) continue;

                Real dx = xi - sx[jt];
                Real dy = yi - sy[jt];
                min_image(dx, dy, Lx, Ly, pbc);

                Real r2 = dx * dx + dy * dy;
                if (r2 < Real(1e-20)) continue;   // skip degenerate pairs

                Real r    = sqrt(r2);
                Real arg  = r * inv_lambda;           // r / λ
                Real fmag = K1_coeff * bessk1_dev(arg);

                Real inv_r = Real(1) / r;
                fxi += fmag * dx * inv_r;   // component along r̂
                fyi += fmag * dy * inv_r;
            }
        }
        __syncthreads();
    }

    // Write drive velocity + ss contribution in one assignment (no prior fill needed)
    if (i < N) {
        vx[i] = init_vx + factor*(fxi + hall_k * fyi);
        vy[i] = init_vy + factor*(fyi - hall_k * fxi);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  F_p — parabolic pinning traps
//
//  Each pin j acts as a harmonic well with maximum force f_p at radius r_p.
//  For particle i at position r_i and pin j at p_j:
//    if |r_i − p_j| < r_p:  F_p += −(f_p / r_p) · (r_i − p_j)
//  Otherwise, the pin exerts no force.  The ratio f_p / r_p is precomputed
//  on the host and passed as an argument to avoid redundant division in the kernel.
// ═════════════════════════════════════════════════════════════════════════════
__global__ static void force_pin_k(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ px,
    const Real* __restrict__ py,
    Real* vx, Real* vy,
    Real f_p_over_rp,   // precomputed f_p / r_p
    Real r_p2,          // r_p²
    Real Lx, Real Ly, Real k, Real factor, bool pbc,
    int N, int N_pins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Real xi  = x[i], yi = y[i];
    Real fxi = Real(0), fyi = Real(0);

    for (int j = 0; j < N_pins; ++j) {
        Real dx = xi - px[j];
        Real dy = yi - py[j];
        min_image(dx, dy, Lx, Ly, pbc);

        Real r2 = dx * dx + dy * dy;
        if (r2 >= r_p2) continue;

        fxi -= f_p_over_rp * dx;   // restoring force toward pin center
        fyi -= f_p_over_rp * dy;
    }

    vx[i] += factor*(fxi + k * fyi);
    vy[i] += factor*(fyi - k * fxi);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Cell-linked list helpers
//
//  point_to_bucket_index — Thrust functor: maps (px, py) → cell index.
//  Cell size equals r_p so the 3×3 neighbourhood always covers all pins
//  within interaction range.
// ═════════════════════════════════════════════════════════════════════════════

struct point_to_bucket_index {
    int  nx, ny;
    Real inv_cell_size;

    __host__ __device__
    int operator()(const thrust::tuple<Real, Real>& pt) const {
        int cx = static_cast<int>(thrust::get<0>(pt) * inv_cell_size);
        int cy = static_cast<int>(thrust::get<1>(pt) * inv_cell_size);
        // clamp handles pins sitting exactly on Lx/Ly
        if (cx >= nx) cx = nx - 1;
        if (cy >= ny) cy = ny - 1;
        if (cx <  0)  cx = 0;
        if (cy <  0)  cy = 0;
        return cy * nx + cx;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  force_pin_cll_k — pinning force using cell-linked list
//  Only the 3×3 neighbour cells of particle i are searched.
// ─────────────────────────────────────────────────────────────────────────────
__global__ static void force_pin_cll_k(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const int*  __restrict__ pin_list,
    const int*  __restrict__ cell_start,
    const int*  __restrict__ cell_end,
    const Real* __restrict__ px,
    const Real* __restrict__ py,
    Real* vx, Real* vy,
    Real f_p_over_rp, Real r_p2,
    Real Lx, Real Ly, Real k, Real factor, bool pbc,
    int cell_nx, int cell_ny,
    Real inv_cell_size,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Real xi = x[i], yi = y[i];
    Real fxi = Real(0), fyi = Real(0);

    // Wrap particle position to [0, box) for cell lookup.
    // Positions are stored unwrapped (unfolded), so we need an explicit wrap.
    Real xi_w, yi_w;
    if (pbc) {
        xi_w = fmod(xi, Lx); if (xi_w < Real(0)) xi_w += Lx;
        yi_w = fmod(yi, Ly); if (yi_w < Real(0)) yi_w += Ly;
    } else {
        xi_w = xi; yi_w = yi;
    }

    int cx = static_cast<int>(xi_w * inv_cell_size);
    int cy = static_cast<int>(yi_w * inv_cell_size);
    if (cx >= cell_nx) cx = cell_nx - 1;
    if (cy >= cell_ny) cy = cell_ny - 1;

    for (int dcy = -1; dcy <= 1; ++dcy) {
        for (int dcx = -1; dcx <= 1; ++dcx) {
            // Periodic cell wrap — for non-PBC the min_image check inside
            // correctly rejects pins that are across the far boundary.
            int ncx = (cx + dcx + cell_nx) % cell_nx;
            int ncy = (cy + dcy + cell_ny) % cell_ny;
            int c   = ncy * cell_nx + ncx;

            int start = cell_start[c];
            int end   = cell_end[c];

            for (int idx = start; idx < end; ++idx) {
                int j = pin_list[idx];
                Real dx = xi - px[j];
                Real dy = yi - py[j];
                min_image(dx, dy, Lx, Ly, pbc);

                Real r2 = dx * dx + dy * dy;
                if (r2 >= r_p2) continue;

                fxi -= f_p_over_rp * dx;
                fyi -= f_p_over_rp * dy;
            }
        }
    }

    vx[i] += factor*(fxi + k * fyi);
    vy[i] += factor*(fyi - k * fxi);
}

// ─────────────────────────────────────────────────────────────────────────────
//  build_pin_cells — host-side one-time setup of the CLL data in SimState
// ─────────────────────────────────────────────────────────────────────────────
void build_pin_cells(SimState& s, const SimParams& p) {
    const int    N_pins    = s.N_pins;
    const double cell_size = p.r_p;           // interaction range == cell size
    const double inv_cs    = 1.0 / cell_size;

    const int nx      = static_cast<int>(std::ceil(skLx(p) / cell_size));
    const int ny      = static_cast<int>(std::ceil(skLy(p) / cell_size));
    const int n_cells = nx * ny;

    s.pin_cell_nx       = nx;
    s.pin_cell_ny       = ny;
    s.pin_inv_cell_size = Real(inv_cs);   // cast for GPU use

    // Allocate CLL arrays
    s.pin_list_cll.resize(N_pins);
    s.cell_start_cll.resize(n_cells);
    s.cell_end_cll.resize(n_cells);

    // 1. pin_list = [0, 1, ..., N_pins-1]
    thrust::sequence(s.pin_list_cll.begin(), s.pin_list_cll.end());

    // 2. Bucket index for each pin
    thrust::device_vector<int> bucket_idx(N_pins);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(s.px.begin(), s.py.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(s.px.end(),   s.py.end())),
        bucket_idx.begin(),
        point_to_bucket_index{nx, ny, Real(inv_cs)});

    // 3. Sort pin indices by bucket (brings spatially close pins together)
    thrust::sort_by_key(bucket_idx.begin(), bucket_idx.end(),
                        s.pin_list_cll.begin());

    // 4. Find the half-open [start, end) of each bucket in pin_list
    thrust::counting_iterator<int> search_begin(0);
    thrust::lower_bound(bucket_idx.begin(), bucket_idx.end(),
                        search_begin, search_begin + n_cells,
                        s.cell_start_cll.begin());
    thrust::upper_bound(bucket_idx.begin(), bucket_idx.end(),
                        search_begin, search_begin + n_cells,
                        s.cell_end_cll.begin());

    s.has_pin_cells = true;
}


// ═════════════════════════════════════════════════════════════════════════════
//  compute_velocities — public interface
//
//  Orchestrates the force contributions in order:
//    1. F_ss + F_D  (all pairs, shared-memory tiled)
//    2. F_p   (skipped if N_pins == 0 or px_d == nullptr)
//
// ═════════════════════════════════════════════════════════════════════════════
void compute_velocities(
    const Real* x_d,  const Real* y_d,
    const Real* px_d, const Real* py_d,
    Real* vx_d, Real* vy_d,
    const SimParams& p,
    const PinCells* pc)
{
    const int  N  = skN(p);
    const Real Lx = Real(skLx(p));
    const Real Ly = Real(skLy(p));
    int blocks = (N + p.n_threads - 1) / p.n_threads;
    const Real vd0_x = v0_x(p);
    const Real vd0_y = v0_y(p);
    const Real k = hall_angle(p);
    const Real factor = inv_denom(p);

    // ── F_ss (initialises vx/vy to drive velocity + ss contribution in one pass)

    // NvtxRange r("force_ss");
    int smem = 2 * p.n_threads * (int)sizeof(Real);
    force_ss_k<<<blocks, p.n_threads, smem>>>(
        x_d, y_d, vx_d, vy_d,
        Real(p.K1), Real(1.0 / p.lambda),
        Lx, Ly, k, factor, vd0_x, vd0_y, p.pbc, N);


    // ── F_p ──────────────────────────────────────────────────────────────────
    if (px_d && p.n_p > 0.0 && p.f_p > 0.0) {
        const Real fpr = Real(p.f_p / p.r_p);
        const Real rp2 = Real(p.r_p * p.r_p);
        if (pc) {
            // NvtxRange r("force_pin_cll");
            force_pin_cll_k<<<blocks, p.n_threads>>>(
                x_d, y_d,
                pc->pin_list, pc->cell_start, pc->cell_end,
                px_d, py_d,
                vx_d, vy_d,
                fpr, rp2,
                Lx, Ly, k, factor, p.pbc,
                pc->nx, pc->ny, pc->inv_cell_size,
                N);
        } else {
            // NvtxRange r("force_pin");
            force_pin_k<<<blocks, p.n_threads>>>(
                x_d, y_d, px_d, py_d,
                vx_d, vy_d,
                fpr, rp2,
                Lx, Ly, k, factor, p.pbc,
                N, skN_pins(p));
        }
    }
}
