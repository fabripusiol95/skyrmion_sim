#pragma once
#include <nvtx3/nvToolsExt.h>

struct NvtxRange {
    NvtxRange(const char* name) { nvtxRangePush(name); }
    ~NvtxRange()                { nvtxRangePop(); }
};
