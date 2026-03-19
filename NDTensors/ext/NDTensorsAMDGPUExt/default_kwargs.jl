using AMDGPU: ROCArray
using NDTensors: NDTensors

NDTensors.default_svd_alg(::Type{<:ROCArray}, a) = "qr_algorithm"
