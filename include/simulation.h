#pragma once
#include "params.h"

namespace simulation {
    // Allocate state, initialise positions & pins, run the time loop,
    // write all output, and return.
    void run(const SimParams& p);
}
