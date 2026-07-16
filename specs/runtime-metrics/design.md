# Design: Runtime Fidelity Metrics in NDTensors

## Problem

`factorize_svd` returned only truncation error. There was no way for callers to inspect how stable the truncation boundary was (gap between last kept and first discarded singular value), and no QR override existed for AMD GPU (the CPU fallback path was used).

## Changes

### 1. Gap Ratio in `Spectrum`

**File:** `NDTensors/src/lib/RankFactorization/src/spectrum.jl`

Added `gap_ratio::ElT` field to `Spectrum`:

```julia
struct Spectrum{VecT, ElT <: Real}
    eigs::VecT
    truncerr::ElT
    gap_ratio::ElT   # s_{χ+1}/s_χ: first discarded / last kept (0 if no truncation)
end
Spectrum(eigs, truncerr) = Spectrum(eigs, truncerr, zero(typeof(truncerr)))  # backward compat
```

**Purpose:** Lets callers know how stable the truncation boundary is. When `gap_ratio` is close to 1, the kept and discarded subspaces are nearly degenerate — the truncation is sensitive to noise in the environment or floating-point errors.

**File:** `NDTensors/src/linearalgebra/linearalgebra.jl`

Gap ratio is computed inside `svd(...)` just before `Spectrum` is constructed:

```julia
_gr = if dS < length(MS)
    MS_cpu = Array(MS[dS:dS+1])   # 2-element CPU copy — works for GPU arrays
    MS_cpu[1] > 0 ? Float64(MS_cpu[2]) / Float64(MS_cpu[1]) : 0.0
else
    0.0
end
spec = Spectrum(P, truncerr, typeof(truncerr)(_gr))
```

`dS` is the number of kept singular values. `MS[dS]` is `s_χ` (last kept), `MS[dS+1]` is `s_{χ+1}` (first discarded). The two-element `Array(...)` copy avoids materializing the full singular value vector on CPU for GPU arrays.

---
