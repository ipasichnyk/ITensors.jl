using AMDGPU: ROCMatrix
using Adapt: adapt
using LinearAlgebra: LinearAlgebra, svd
using NDTensors.AMDGPUExtensions: roc
using NDTensors.Expose: Expose, Exposed, expose, unexpose, ql, ql_positive
using NDTensors.GPUArraysCoreExtensions: cpu
using NDTensors.Vendored.TypeParameterAccessors: unwrap_array_type

function LinearAlgebra.svd(A::Exposed{<:ROCMatrix}; kwargs...)
    return svd(unexpose(A); alg=LinearAlgebra.QRIteration())
end

## TODO currently AMDGPU doesn't have ql so make a ql function
function Expose.ql(A::Exposed{<:ROCMatrix})
    Q, L = ql(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
function Expose.ql_positive(A::Exposed{<:ROCMatrix})
    Q, L = ql_positive(expose(cpu(A)))
    return adapt(unwrap_array_type(A), Matrix(Q)), adapt(unwrap_array_type(A), L)
end
