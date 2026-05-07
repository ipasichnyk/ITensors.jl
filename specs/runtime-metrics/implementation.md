# Implementation: Runtime Fidelity Metrics (NDTensors)

## Status: Done

## Implemented

- [x] **`gap_ratio` field in `Spectrum`** — `NDTensors/src/lib/RankFactorization/src/spectrum.jl`
  - Backward-compatible two-arg constructor preserved
  - `gap_ratio(s::Spectrum)` accessor added
  - Exported from `NDTensors/src/exports.jl`
- [x] **Gap ratio computed in `svd`** — `NDTensors/src/linearalgebra/linearalgebra.jl`
  - `Array(MS[dS:dS+1])` two-element CPU copy avoids full vector transfer for GPU arrays
  - Stored as `typeof(truncerr)(_gr)` to match precision of the truncation error
## Files Changed

| File | Change |
|------|--------|
| `NDTensors/src/lib/RankFactorization/src/spectrum.jl` | Added `gap_ratio` field and accessor |
| `NDTensors/src/linearalgebra/linearalgebra.jl` | Computes and stores gap ratio in `svd` |
| `NDTensors/src/exports.jl` | Exports `gap_ratio` |

