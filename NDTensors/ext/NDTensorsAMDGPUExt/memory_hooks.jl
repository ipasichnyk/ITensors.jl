function __init__()
    # Register a reclaim hook so AMDGPU.jl's alloc_or_retry! can nudge Julia GC
    # when GPU memory is exhausted (fires at phase 6 — last resort before OOM).
    if isdefined(AMDGPU, :reclaim_hooks)
        push!(AMDGPU.reclaim_hooks, () -> (GC.gc(false); nothing))
    end
end
