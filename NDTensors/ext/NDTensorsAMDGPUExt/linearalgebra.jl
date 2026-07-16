using AMDGPU: AMDGPU, ROCMatrix
using Adapt: adapt
using LinearAlgebra: LinearAlgebra, SVD, svd
using NDTensors.Expose: Expose, Exposed, expose, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using TypeParameterAccessors: unwrap_array_type
using NDTensors: NDTensors

function NDTensors.svd_catch_error(A::ROCMatrix; alg::String="qr_algorithm")
    if alg == "qr_algorithm"
        return _svd_qr(A)
    elseif alg == "jacobi_algorithm"
        return _svd_jacobi(A)
    else
        error("svd algorithm $alg is not currently supported for ROCMatrix. Use \"qr_algorithm\" or \"jacobi_algorithm\".")
    end
end

function _svd_qr(A::ROCMatrix)
    USV = try
        svd(A; alg=LinearAlgebra.QRIteration())
    catch
        return nothing
    end
    return USV
end

function _svd_jacobi(A::ROCMatrix)
    USV = try
        abstol = zero(real(eltype(A)))
        max_sweeps = Cint(100)
        # gesvdj! returns (U, S, V', ...) where V is (k×n) — V' is (n×k).
        # SVD expects Vt of shape (k×n), so we adjoint again.
        U, S, V, residual, n_sweeps, info = AMDGPU.rocSOLVER.gesvdj!(copy(A), abstol, max_sweeps)
        SVD(U, S, V')
    catch
        return nothing
    end
    return USV
end

function Expose.ql(A::Exposed{<:ROCMatrix})
    Q, L = ql(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function Expose.ql_positive(A::Exposed{<:ROCMatrix})
    Q, L = ql_positive(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
