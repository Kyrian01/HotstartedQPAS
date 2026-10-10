"""
    Cholmod

Thin Julia layer over the CHOLMOD library of SuiteSparse, restricted to what the hot-started
QPAS needs:

- a simplicial LDLᵀ factor in natural ordering that starts as the identity,
- in-place symmetric row/column modifications with `cholmod_l_rowadd` / `cholmod_l_rowdel`
  (thesis §4.4.3),
- repeated solves with `cholmod_l_solve2`, whose workspace is allocated once and reused.

File layout:

| file              | contents                                                          |
|:------------------|:------------------------------------------------------------------|
| `types.jl`        | constants and Julia mirrors of the C structs                      |
| `libcholmod.jl`   | raw `ccall` bindings, the only place that calls into CHOLMOD      |
| `common.jl`       | owned `cholmod_common` workspace                                  |
| `ldlt_factor.jl`  | [`LDLtFactor`](@ref), the object the QPAS works with              |
"""
module Cholmod

using SuiteSparse_jll: libcholmod
import SparseArrays
using SparseArrays: SparseMatrixCSC, sparse

include("types.jl")
include("libcholmod.jl")
include("common.jl")
include("ldlt_factor.jl")

export LDLtFactor, CholmodError
export rowadd!, rowdel!, solve!, rhs_buffer, pivot, min_scaled_pivot, reset!, dimension
export factor_matrices
export cholmod_version

function __init__()
    check_struct_layouts()
end

end # module Cholmod
