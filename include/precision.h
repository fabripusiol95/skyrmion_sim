#pragma once
// Compile with -DSK_FLOAT to use single precision on the GPU.
// All GPU arrays, kernel arguments, and device math use Real.
// Host-side time tracking (SimState::t, dt_current) stays double.
#ifdef SK_FLOAT
    using Real = float;
#else
    using Real = double;
#endif
