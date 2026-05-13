#include "params.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string trim(const std::string& s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

// Remove inline comment starting with '#' or ';'
static std::string strip_comment(const std::string& s) {
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '#' || s[i] == ';') return s.substr(0, i);
    }
    return s;
}

static bool parse_bool(const std::string& val, const std::string& key) {
    std::string v = to_lower(trim(val));
    if (v == "1" || v == "true"  || v == "yes" || v == "on")  return true;
    if (v == "0" || v == "false" || v == "no"  || v == "off") return false;
    throw std::runtime_error("Invalid boolean value for key '" + key + "': " + val);
}

static Integrator parse_integrator(const std::string& val) {
    std::string v = to_lower(trim(val));
    if (v == "euler") return Integrator::EULER;
    if (v == "rk2")   return Integrator::RK2;
    throw std::runtime_error("Unknown integrator '" + val +
                             "'. Valid: euler, rk2");
}

static const char* integrator_name(Integrator ig) {
    switch (ig) {
        case Integrator::EULER: return "euler";
        case Integrator::RK2:   return "rk2";
    }
    return "unknown";
}

static InitLayout parse_init_layout(const std::string& val) {
    std::string v = to_lower(trim(val));
    if (v == "triangular") return InitLayout::TRIANGULAR;
    if (v == "random")     return InitLayout::RANDOM;
    if (v == "file")       return InitLayout::FILE;
    throw std::runtime_error("Unknown init_layout '" + val +
                             "'. Valid: triangular, random, file");
}

// ─────────────────────────────────────────────────────────────────────────────
//  parse_params
// ─────────────────────────────────────────────────────────────────────────────
SimParams parse_params(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open parameter file: " + filename);

    SimParams p;

    // Track which keys were set (for duplicate detection)
    std::unordered_set<std::string> seen;

    int lineno = 0;
    std::string line;

    while (std::getline(f, line)) {
        ++lineno;
        line = strip_comment(line);
        line = trim(line);

        if (line.empty()) continue;          // blank / comment-only line
        if (line[0] == '[') continue;        // INI section header — ignored

        auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("Line " + std::to_string(lineno) +
                                     ": missing '=' in '" + line + "'");

        std::string key = to_lower(trim(line.substr(0, eq)));
        std::string val = trim(strip_comment(line.substr(eq + 1)));

        if (seen.count(key))
            throw std::runtime_error("Line " + std::to_string(lineno) +
                                     ": duplicate key '" + key + "'");
        seen.insert(key);

        // ── System ──────────────────────────────────────────────────────────
        if      (key == "nx")           p.nx          = std::stoi(val);
        else if (key == "ny")           p.ny          = std::stoi(val);
        else if (key == "n_sk")         p.n_sk        = std::stod(val);
        else if (key == "pbc")          p.pbc         = parse_bool(val, key);

        // ── Thiele ──────────────────────────────────────────────────────────
        else if (key == "alpha")        p.alpha       = std::stod(val);
        else if (key == "g_z")          p.G_z         = std::stod(val);

        // ── Skyrmion–skyrmion ────────────────────────────────────────────────
        else if (key == "k1")           p.K1          = std::stod(val);
        else if (key == "lambda")       p.lambda      = std::stod(val);

        // ── Pinning ──────────────────────────────────────────────────────────
        else if (key == "n_p")          p.n_p         = std::stod(val);
        else if (key == "f_p")          p.f_p         = std::stod(val);
        else if (key == "r_p")          p.r_p         = std::stod(val);
        else if (key == "pin_file")     p.pin_file    = val;

        // ── Drive ────────────────────────────────────────────────────────────
        else if (key == "f_d")          p.F_D         = std::stod(val);
        else if (key == "drive_angle")  p.drive_angle = std::stod(val);

        // ── Integration ──────────────────────────────────────────────────────
        else if (key == "integrator")   p.integrator  = parse_integrator(val);
        else if (key == "dt")           p.dt          = std::stod(val);
        else if (key == "t_max")        p.t_max       = std::stod(val);

        // ── Initial configuration ────────────────────────────────────────────
        else if (key == "init_layout")  p.init_layout = parse_init_layout(val);
        else if (key == "init_file")    p.init_file   = val;
        else if (key == "seed")         p.seed        = std::stoull(val);

        // ── Output ───────────────────────────────────────────────────────────
        else if (key == "output_dir")   p.output_dir  = val;
        else if (key == "run_tag")      p.run_tag     = val;
        else if (key == "save_every")   p.save_every  = std::stoi(val);
        else if (key == "save_vel")     p.save_vel    = parse_bool(val, key);
        else if (key == "save_forces")  p.save_forces = parse_bool(val, key);
        else if (key == "thermo_every") p.thermo_every= std::stoi(val);

        // ── Misc ─────────────────────────────────────────────────────────────
        else if (key == "n_threads")      p.n_threads     = std::stoi(val);
        else if (key == "verbose")        p.verbose       = parse_bool(val, key);
        else if (key == "use_pin_cells")  p.use_pin_cells = parse_bool(val, key);

        else
            throw std::runtime_error("Line " + std::to_string(lineno) +
                                     ": unknown key '" + key + "'");
    }

    // ── Validation ────────────────────────────────────────────────────────────
    if (p.nx <= 0 || p.ny <= 0)
        throw std::runtime_error("nx and ny must be positive");
    if (p.n_sk <= 0)
        throw std::runtime_error("n_sk must be positive");
    if (p.alpha <= 0)
        throw std::runtime_error("alpha must be positive");
    if (p.lambda <= 0)
        throw std::runtime_error("lambda must be positive");
    if (p.dt <= 0)
        throw std::runtime_error("dt must be positive");
    if (p.t_max <= 0)
        throw std::runtime_error("t_max must be positive");
    if (p.init_layout == InitLayout::FILE && p.init_file.empty())
        throw std::runtime_error("init_layout = file requires init_file to be set");
    if (p.n_p < 0.0)
        throw std::runtime_error("n_p must be non-negative");
    // n_p > 0 with no pin_file → random placement, OK

    if (p.verbose) {
        std::cout << "[params] Loaded " << filename << "\n"
                  << "  N=" << skN(p) << "  L=(" << skLx(p) << "," << skLy(p) << ")\n"
                  << "  alpha=" << p.alpha << "  G_z=" << p.G_z
                  << "  k=" << p.G_z/p.alpha << "\n"
                  << "  integrator=" << integrator_name(p.integrator) << "  dt=" << p.dt
                  << "  t_max=" << p.t_max << "\n";
    }

    return p;
}
