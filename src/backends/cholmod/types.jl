# Constants and memory-layout mirrors of the CHOLMOD C structs (cholmod.h, CHOLMOD ≥ 5).
#
# The mirrors are only ever *read* (via `unsafe_load`) or pointed at; CHOLMOD owns the memory.
# `check_struct_layouts` verifies them against the bindings generated for the exact library
# version shipped with Julia, so an incompatible SuiteSparse upgrade fails loudly at load time.

"`SuiteSparse_long`: the index type of the `cholmod_l_*` API (int64 on 64-bit platforms)."
const SuiteSparseLong = Int64

const CHOLMOD_TRUE = Cint(1)
const CHOLMOD_FALSE = Cint(0)

# xtype / dtype (combined as `xdtype = xtype + dtype` in CHOLMOD 5)
const CHOLMOD_PATTERN = Cint(0)
const CHOLMOD_REAL = Cint(1)
const CHOLMOD_DOUBLE = Cint(0)
const CHOLMOD_REAL_DOUBLE = CHOLMOD_REAL + CHOLMOD_DOUBLE

# `sys` argument of cholmod_solve2
const CHOLMOD_A = Cint(0)

# Common->status codes
const CHOLMOD_OK = Cint(0)
const CHOLMOD_NOT_POSDEF = Cint(1)

"Mirror of `cholmod_dense`."
struct CholmodDense
    nrow::Csize_t
    ncol::Csize_t
    nzmax::Csize_t
    d::Csize_t
    x::Ptr{Cdouble}
    z::Ptr{Cvoid}
    xtype::Cint
    dtype::Cint
end

"Mirror of `cholmod_sparse`."
struct CholmodSparse
    nrow::Csize_t
    ncol::Csize_t
    nzmax::Csize_t
    p::Ptr{SuiteSparseLong}
    i::Ptr{SuiteSparseLong}
    nz::Ptr{SuiteSparseLong}
    x::Ptr{Cdouble}
    z::Ptr{Cvoid}
    stype::Cint
    itype::Cint
    xtype::Cint
    dtype::Cint
    sorted::Cint
    packed::Cint
end

"Mirror of `cholmod_factor`. For a simplicial LDLᵀ factor, column `j` of `L` is stored in
`i[p[j]+1 : p[j]+nz[j]]` / `x[...]` (0-based), with the diagonal `D[j]` as its first entry."
struct CholmodFactor
    n::Csize_t
    minor::Csize_t
    Perm::Ptr{SuiteSparseLong}
    ColCount::Ptr{SuiteSparseLong}
    IPerm::Ptr{SuiteSparseLong}
    nzmax::Csize_t
    p::Ptr{SuiteSparseLong}
    i::Ptr{SuiteSparseLong}
    x::Ptr{Cdouble}
    z::Ptr{Cvoid}
    nz::Ptr{SuiteSparseLong}
    next::Ptr{SuiteSparseLong}
    prev::Ptr{SuiteSparseLong}
    nsuper::Csize_t
    ssize::Csize_t
    xsize::Csize_t
    maxcsize::Csize_t
    maxesize::Csize_t
    super::Ptr{SuiteSparseLong}
    pi::Ptr{SuiteSparseLong}
    px::Ptr{SuiteSparseLong}
    s::Ptr{SuiteSparseLong}
    ordering::Cint
    is_ll::Cint
    is_super::Cint
    is_monotonic::Cint
    itype::Cint
    xtype::Cint
    dtype::Cint
    useGPU::Cint
end

"""
    check_struct_layouts()

Compare the mirrors above with the bindings that SparseArrays generated for the bundled
CHOLMOD, and throw if any field offset or struct size differs.
"""
function check_struct_layouts()
    LSS = SparseArrays.LibSuiteSparse
    pairs = (
        CholmodDense => LSS.cholmod_dense_struct,
        CholmodSparse => LSS.cholmod_sparse_struct,
        CholmodFactor => LSS.cholmod_factor_struct,
    )
    for (mine, ref) in pairs
        sizeof(mine) == sizeof(ref) ||
            error("CHOLMOD layout mismatch: sizeof($mine)=$(sizeof(mine)) vs $(sizeof(ref))")
        fieldnames(mine) == fieldnames(ref) ||
            error("CHOLMOD layout mismatch: field names of $mine differ from $ref")
        for k in 1:fieldcount(mine)
            fieldoffset(mine, k) == fieldoffset(ref, k) ||
                error("CHOLMOD layout mismatch: offset of $mine.$(fieldname(mine, k))")
        end
    end
    return nothing
end
