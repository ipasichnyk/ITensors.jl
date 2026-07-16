"""
Spectrum
contains the (truncated) density matrix eigenvalue spectrum which is computed during a
decomposition done by `svd` or `eigen`. In addition stores the truncation error.
"""
struct Spectrum{VecT <: Union{AbstractVector, Nothing}, ElT <: Real}
    eigs::VecT
    truncerr::ElT
    gap_ratio::ElT   # s_{χ+1}/s_χ: first discarded / last kept singular value (0 if no truncation)
end

# Backward-compatible constructor without gap_ratio
Spectrum(eigs, truncerr) = Spectrum(eigs, truncerr, zero(typeof(truncerr)))

eigs(s::Spectrum) = s.eigs
truncerror(s::Spectrum) = s.truncerr
gap_ratio(s::Spectrum) = s.gap_ratio

function entropy(s::Spectrum)
    S = 0.0
    eigs_s = eigs(s)
    isnothing(eigs_s) &&
        error("Spectrum does not contain any eigenvalues, cannot compute the entropy")
    for p in eigs_s
        p > 1.0e-13 && (S -= p * log(p))
    end
    return S
end
