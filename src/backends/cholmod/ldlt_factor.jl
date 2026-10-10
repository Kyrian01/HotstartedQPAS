# `LDLtFactor`: fixed-size, updatable LDLᵀ factorization backed by CHOLMOD.
#
# Thesis §4.4.3 / §4.5.1: the reduced Hessian is stored as an n_max × n_max symmetric matrix that
# equals the identity on all inactive indices. Activating index k (moving it into the nullspace
# set N) is a `rowadd`, deactivating it is a `rowdel`. The factor uses CHOLMOD's natural ordering
# (no fill-reducing permutation), so index k of the factor is index k of the reduced Hessian.
#
# All CHOLMOD-owned arrays that are touched in the hot path (the right-hand side B, the solution
# X, the row buffer R) are exposed as Julia `Vector`s that alias CHOLMOD memory, so there are no
# copies and no allocations per update or solve.

"""
    LDLtFactor(n)

Simplicial LDLᵀ factorization of a symmetric `n × n` matrix, initialised to the identity.

Main operations:
- [`rowadd!`](@ref)`(F, k, rows, vals, nnz)`: set row/column `k` (which must currently be an
  identity row) to the given sparse vector and update the factor;
- [`rowdel!`](@ref)`(F, k)`: reset row/column `k` to the identity and update the factor;
- [`solve!`](@ref)`(F)`: solve with the right-hand side stored in [`rhs_buffer`](@ref)`(F)`;
- [`solve!`](@ref)`(F, pattern, npattern)`: the same for a right-hand side that is zero
  outside `pattern`, at a cost independent of `n`;
- [`pivot`](@ref)`(F, k)`: the diagonal entry `D[k]`.

The object owns its CHOLMOD memory and frees it in a finalizer or in `close`.
"""
mutable struct LDLtFactor
    n::Int
    common::CholmodCommon
    L::Ptr{CholmodFactor}
    # solve2 handles (allocated on first solve, then reused)
    B::Ptr{CholmodDense}
    X::Base.RefValue{Ptr{CholmodDense}}
    Y::Base.RefValue{Ptr{CholmodDense}}
    E::Base.RefValue{Ptr{CholmodDense}}
    # n × 1 sparse column passed to rowadd
    R::Ptr{CholmodSparse}
    # n × 1 pattern of a sparse right-hand side, and the solution pattern returned by solve2
    Bset::Ptr{CholmodSparse}
    Xset::Base.RefValue{Ptr{CholmodSparse}}
    # Julia views on CHOLMOD memory
    rhs::Vector{Float64}
    sol::Vector{Float64}
    R_p::Vector{SuiteSparseLong}
    R_i::Vector{SuiteSparseLong}
    R_x::Vector{Float64}
    Bset_p::Vector{SuiteSparseLong}
    Bset_i::Vector{SuiteSparseLong}
    closed::Bool
end

function LDLtFactor(n::Integer)
    n > 0 || throw(ArgumentError("LDLtFactor dimension must be positive, got $n"))
    common = CholmodCommon()
    cp = common_ptr(common)

    L = _identity_factor(n, common)

    B = cholmod_l_allocate_dense(n, 1, n, CHOLMOD_REAL_DOUBLE, cp)
    B == C_NULL && throw(CholmodError("cholmod_l_allocate_dense", status(common)))
    rhs = unsafe_wrap(Array, unsafe_load(B).x, n; own = false)
    fill!(rhs, 0.0)

    R = cholmod_l_allocate_sparse(n, 1, n, CHOLMOD_FALSE, CHOLMOD_TRUE, Cint(0),
                                  CHOLMOD_REAL_DOUBLE, cp)
    R == C_NULL && throw(CholmodError("cholmod_l_allocate_sparse", status(common)))
    Rh = unsafe_load(R)
    R_p = unsafe_wrap(Array, Rh.p, 2; own = false)
    R_i = unsafe_wrap(Array, Rh.i, n; own = false)
    R_x = unsafe_wrap(Array, Rh.x, n; own = false)
    R_p[1] = 0
    R_p[2] = 0

    Bset = cholmod_l_allocate_sparse(n, 1, n, CHOLMOD_FALSE, CHOLMOD_TRUE, Cint(0),
                                     CHOLMOD_PATTERN, cp)
    Bset == C_NULL && throw(CholmodError("cholmod_l_allocate_sparse", status(common)))
    Bh = unsafe_load(Bset)
    Bset_p = unsafe_wrap(Array, Bh.p, 2; own = false)
    Bset_i = unsafe_wrap(Array, Bh.i, n; own = false)
    Bset_p[1] = 0
    Bset_p[2] = 0

    F = LDLtFactor(Int(n), common, L, B,
                   Ref{Ptr{CholmodDense}}(C_NULL), Ref{Ptr{CholmodDense}}(C_NULL),
                   Ref{Ptr{CholmodDense}}(C_NULL), R, Bset, Ref{Ptr{CholmodSparse}}(C_NULL),
                   rhs, Float64[], R_p, R_i, R_x, Bset_p, Bset_i, false)
    finalizer(close, F)
    return F
end

function _identity_factor(n::Integer, common::CholmodCommon)
    cp = common_ptr(common)
    L = cholmod_l_allocate_factor(n, cp)
    L == C_NULL && throw(CholmodError("cholmod_l_allocate_factor", status(common)))
    # symbolic → numeric simplicial LDLᵀ: CHOLMOD initialises a symbolic factor to L = D = I.
    ret = cholmod_l_change_factor(CHOLMOD_REAL, CHOLMOD_FALSE, CHOLMOD_FALSE, CHOLMOD_FALSE,
                                  CHOLMOD_TRUE, L, cp)
    if ret != CHOLMOD_TRUE
        cholmod_l_free_factor(Ref(L), cp)
        throw(CholmodError("cholmod_l_change_factor", status(common)))
    end
    return L
end

"Dimension of the factorized matrix."
dimension(F::LDLtFactor) = F.n

"""
    rhs_buffer(F) -> Vector{Float64}

Right-hand side vector used by [`solve!`](@ref). It aliases CHOLMOD memory: write into it in
place, never replace it.
"""
rhs_buffer(F::LDLtFactor) = F.rhs

"""
    rowadd!(F, k, rows, vals, nnz)

Set row and column `k` of the factorized matrix to the sparse vector given by the first `nnz`
entries of `rows` (1-based, any order, must include `k` itself) and `vals`, and update the
factorization in place. Row/column `k` must currently be an identity row.

Returns `true` if CHOLMOD reports the updated matrix as numerically positive definite.
"""
function rowadd!(F::LDLtFactor, k::Integer, rows::AbstractVector{<:Integer},
                 vals::AbstractVector{Float64}, nnz::Integer)
    _check_open(F)
    1 <= k <= F.n || throw(BoundsError(F, k))
    nnz <= F.n || throw(ArgumentError("rowadd!: nnz = $nnz exceeds dimension $(F.n)"))
    R_i, R_x = F.R_i, F.R_x
    @inbounds for t in 1:nnz
        R_i[t] = rows[t] - 1
        R_x[t] = vals[t]
    end
    F.R_p[2] = nnz
    cp = common_ptr(F.common)
    GC.@preserve F check_call(cholmod_l_rowadd(k - 1, F.R, F.L, cp), F.common, "cholmod_l_rowadd")
    return status(F.common) != CHOLMOD_NOT_POSDEF
end

"""
    rowdel!(F, k)

Reset row and column `k` of the factorized matrix to the identity and update the
factorization in place.
"""
function rowdel!(F::LDLtFactor, k::Integer)
    _check_open(F)
    1 <= k <= F.n || throw(BoundsError(F, k))
    cp = common_ptr(F.common)
    GC.@preserve F check_call(cholmod_l_rowdel(k - 1, Ptr{CholmodSparse}(C_NULL), F.L, cp),
                              F.common, "cholmod_l_rowdel")
    return nothing
end

"""
    solve!(F) -> Vector{Float64}

Solve `(L D Lᵀ) x = b` with `b = rhs_buffer(F)`. The returned vector aliases CHOLMOD's solution
buffer and is overwritten by the next solve. `rhs_buffer(F)` is not modified.
"""
function solve!(F::LDLtFactor)
    _check_open(F)
    cp = common_ptr(F.common)
    GC.@preserve F check_call(cholmod_l_solve2(CHOLMOD_A, F.L, F.B, Ptr{CholmodSparse}(C_NULL),
                                               F.X, Ref{Ptr{CholmodSparse}}(C_NULL), F.Y, F.E, cp),
                              F.common, "cholmod_l_solve2")
    return _solution_view(F)
end

"""
    solve!(F, pattern, npattern) -> Vector{Float64}

Sparse right-hand-side solve: like `solve!(F)`, but `rhs_buffer(F)` is only read at the
first `npattern` indices of `pattern` (1-based) and is treated as zero elsewhere. The
solution is computed only on the reach of `pattern` in the factor, which contains
`pattern`; entries of the returned vector outside that reach are not defined.
Cost is independent of the dimension `n` (CHOLMOD `solve2` with `Bset`).
"""
function solve!(F::LDLtFactor, pattern::AbstractVector{<:Integer}, npattern::Integer)
    _check_open(F)
    npattern <= F.n || throw(ArgumentError("pattern longer than the dimension"))
    Bset_i = F.Bset_i
    @inbounds for t in 1:npattern
        Bset_i[t] = pattern[t] - 1
    end
    F.Bset_p[2] = npattern
    cp = common_ptr(F.common)
    GC.@preserve F check_call(cholmod_l_solve2(CHOLMOD_A, F.L, F.B, F.Bset, F.X, F.Xset,
                                               F.Y, F.E, cp),
                              F.common, "cholmod_l_solve2")
    return _solution_view(F)
end

# Julia view on CHOLMOD's solution buffer (re-wrapped only if CHOLMOD reallocated it).
function _solution_view(F::LDLtFactor)
    xptr = unsafe_load(F.X[]).x
    if length(F.sol) != F.n || pointer(F.sol) != xptr
        F.sol = unsafe_wrap(Array, xptr, F.n; own = false)
    end
    return F.sol
end

"""
    pivot(F, k) -> Float64

Diagonal entry `D[k]` of the LDLᵀ factorization.
"""
function pivot(F::LDLtFactor, k::Integer)
    _check_open(F)
    1 <= k <= F.n || throw(BoundsError(F, k))
    f = unsafe_load(F.L)
    first = unsafe_load(f.p, k)              # 0-based start of column k
    return unsafe_load(f.x, first + 1)       # diagonal is stored first
end

"""
    min_scaled_pivot(F, indices, scale) -> Float64

`min D[k] / scale[k]` over `k ∈ indices` (`Inf` if empty), reading the factor header once.
"""
function min_scaled_pivot(F::LDLtFactor, indices::AbstractVector{<:Integer},
                          scale::AbstractVector{Float64})
    _check_open(F)
    f = unsafe_load(F.L)
    m = Inf
    @inbounds for k in indices
        d = unsafe_load(f.x, unsafe_load(f.p, k) + 1)
        m = min(m, d / max(scale[k], floatmin(Float64)))
    end
    return m
end

"""
    reset!(F)

Discard the current factorization and reinitialise `F` to the identity.
"""
function reset!(F::LDLtFactor)
    _check_open(F)
    cp = common_ptr(F.common)
    GC.@preserve F begin
        cholmod_l_free_factor(Ref(F.L), cp)
        F.L = _identity_factor(F.n, F.common)
    end
    return F
end

"""
    factor_matrices(F) -> (L::SparseMatrixCSC, d::Vector{Float64})

Copy the factorization out as a unit lower-triangular `L` and diagonal `d`, so that the
factorized matrix equals `L * Diagonal(d) * L'`. Intended for debugging and tests.
"""
function factor_matrices(F::LDLtFactor)
    _check_open(F)
    f = unsafe_load(F.L)
    n = F.n
    Is = Int[]
    Js = Int[]
    Vs = Float64[]
    d = zeros(n)
    for j in 1:n
        start = unsafe_load(f.p, j)
        cnt = unsafe_load(f.nz, j)
        for t in start:(start + cnt - 1)
            i = unsafe_load(f.i, t + 1) + 1
            v = unsafe_load(f.x, t + 1)
            if i == j
                d[j] = v
            else
                push!(Is, i)
                push!(Js, j)
                push!(Vs, v)
            end
        end
        push!(Is, j)
        push!(Js, j)
        push!(Vs, 1.0)
    end
    return sparse(Is, Js, Vs, n, n), d
end

"Free all CHOLMOD memory owned by `F`. Called automatically by the finalizer."
function Base.close(F::LDLtFactor)
    F.closed && return nothing
    F.closed = true
    cp = common_ptr(F.common)
    GC.@preserve F begin
        F.sol = Float64[]
        F.rhs = Float64[]
        cholmod_l_free_factor(Ref(F.L), cp)
        cholmod_l_free_dense(Ref(F.B), cp)
        for h in (F.X, F.Y, F.E)
            h[] != C_NULL && cholmod_l_free_dense(h, cp)
        end
        cholmod_l_free_sparse(Ref(F.R), cp)
        cholmod_l_free_sparse(Ref(F.Bset), cp)
        F.Xset[] != C_NULL && cholmod_l_free_sparse(F.Xset, cp)
        finish!(F.common)
    end
    return nothing
end

@inline _check_open(F::LDLtFactor) =
    F.closed && throw(ArgumentError("LDLtFactor has been closed"))

Base.show(io::IO, F::LDLtFactor) =
    print(io, "LDLtFactor(", F.n, "×", F.n, F.closed ? ", closed)" : ")")
