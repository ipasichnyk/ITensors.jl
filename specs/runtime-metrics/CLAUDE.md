# CLAUDE.md — Runtime Fidelity Metrics (NDTensors)

## Project Context
This spec covers changes to NDTensors that expose per-truncation fidelity metrics to callers, and the QR override for AMD GPU. These changes support runtime diagnostics in TensorNetworkQuantumSimulator.jl.

## Before Starting Work
1. Read [specs/runtime-metrics/design.md](design.md) — what changed and why
2. Check [specs/runtime-metrics/implementation.md](implementation.md) — current status
3. Key files: `NDTensors/src/lib/RankFactorization/src/spectrum.jl`, `NDTensors/src/linearalgebra/linearalgebra.jl`, `NDTensors/src/exports.jl`

## Code Patterns
- `Spectrum` has a backward-compatible constructor `Spectrum(eigs, truncerr)` — existing callers not broken
- GPU array values must be pulled to CPU scalars before arithmetic (use `Array(MS[i:i+1])`)

## Don't
- Don't break the `Spectrum(eigs, truncerr)` two-arg constructor — used widely in NDTensors
- Don't add GPU-specific logic outside the AMDGPU extension
