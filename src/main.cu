#include <iostream>
#include <stdexcept>
#include "params.h"
#include "simulation.h"
#include <nvtx3/nvToolsExt.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: skyrmion_sim <params.conf>\n";
        return 1;
    }

    SimParams p;
    try {
        p = parse_params(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }

    std::cout << "[skyrmion_sim]"
              << "  N="      << skN(p)
              << "  L=("     << skLx(p) << "x" << skLy(p) << ")"
              << "  k="      << hall_angle(p)
              << "  t_max="  << p.t_max
              << "\n";

    try {
        simulation::run(p);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}