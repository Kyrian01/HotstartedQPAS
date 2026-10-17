# `LDLtFactor`: fixed-size, updatable LDLᵀ factorization backed by CHOLMOD.
#
# Thesis §6.4 (Updating the LDLᵀ factorization) / §6.1 (Fixed-length representation, Def. 6.1.1):
# the (effective) reduced Hessian is stored as an n_max × n_max symmetric matrix that
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
mutable struct LDLtFactor <: UpdatableLDLt
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
    # n × 1 sparse column passed to updown (row indices sorted, as CHOLMOD requires there)
    C::Ptr{CholmodSparse}
    # n × 1 pattern of a sparse right-hand side, and the solution pattern returned by solve2
    Bset::Ptr{CholmodSparse}
    Xset::Base.RefValue{Ptr{CholmodSparse}}
    # Julia views on CHOLMOD memory
    rhs::Vector{Float64}
    sol::Vector{Float64}
    R_p::Vector{SuiteSparseLong}
    R_i::Vector{SuiteSparseLong}
    R_x::Vector{Float64}
    C_p::Vector{SuiteSparseLong}
    C_i::Vector{SuiteSparseLong}
    C_x::Vector{Float64}
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

    C = cholmod_l_allocate_sparse(n, 1, n, CHOLMOD_TRUE, CHOLMOD_TRUE, Cint(0),
                                  CHOLMOD_REAL_DOUBLE, cp)
    C == C_NULL && throw(CholmodError("cholmod_l_allocate_sparse", status(common)))
    Ch = unsafe_load(C)
    C_p = unsafe_wrap(Array, Ch.p, 2; own = false)
    C_i = unsafe_wrap(Array, Ch.i, n; own = false)
    C_x = unsafe_wrap(Array, Ch.x, n; own = false)
    C_p[1] = 0
    C_p[2] = 0

    Bset =cholmod_l_allocate_sparse(n, 1, n, CHOLMOD_FALSE, CHOLMOD_TRUE, Cint(0),
                                     CHOLMOD_PATTERN, cp)
    Bset == C_NULL && throw(CholmodError("cholmod_l_allocate_sparse", status(common)))
    Bh = unsafe_load(Bset)
    Bset_p = unsafe_wrap(Array, Bh.p, 2; own = false)
    Bset_i = unsafe_wrap(Array, Bh.i, n; own = false)
    Bset_p[1] = 0
    Bset_p[2] = 0

    F = LDLtFactor(Int(n), common, L, B,
                   Ref{Ptr{CholmodDense}}(C_NULL), Ref{Ptr{CholmodDense}}(C_NULL),
                   Ref{Ptr{CholmodDense}}(C_NULL), R, C, Bset, Ref{Ptr{CholmodSparse}}(C_NULL),
                   rhs, Float64[], R_p, R_i, R_x, C_p, C_i, C_x, Bset_p, Bset_i, false)
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

backend_name(::LDLtFactor) = "cholmod"

# rowdel / updown zero entries numerically but keep them in the pattern of L
needs_periodic_refactor(::LDLtFactor) = true

backend_stats(::LDLtFactor) = NamedTuple()

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

Returns `true` unless CHOLMOD flags the result as not positive definite. In practice
`cholmod_l_rowadd` does not check definiteness (a negative new pivot is returned with `true`),
so callers must test the pivots themselves (`min_scaled_pivot`).
"""
function rowadd!(F::LDLtFactor, k::Integer, rows::AbstractVector{<:Integer},
                 vals::AbstractVector{Float64}, nnz::Integer)
    _check_open(F)
    1 <= k <= F.n || throw(BoundsError(F, k))
    nnz <= F.n || throw(ArgumentError("rowadd!: nnz = $nnz exceeds dimension $(F.n)"))
    # copy the new row/column into CHOLMOD's n × 1 sparse buffer R (0-based row indices)
    R_i, R_x = F.R_i, F.R_x
    @inbounds for t in 1:nnz
        R_i[t] = rows[t] - 1
        R_x[t] = vals[t]
    end
    F.R_p[2] = nnz
    cp = common_ptr(F.common)
    # row k of L: triangular solve with the leading block; trailing block: rank-one downdate
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
    # trailing block: rank-one update d_k ℓℓᵀ, then row/column k of L and D become e_k / 1
    GC.@preserve F check_call(cholmod_l_rowdel(k - 1, Ptr{CholmodSparse}(C_NULL), F.L, cp),
                              F.common, "cholmod_l_rowdel")
    return nothing
end

"""
    updown!(F, update, rows, vals, nnz) -> Bool

Rank-one modification of the factorized matrix: `M ← M + vvᵀ` if `update`, `M ← M - vvᵀ`
otherwise, where `v` is the sparse vector given by the first `nnz` entries of `rows`
(1-based, any order, no duplicates) and `vals`. The factor is updated in place.

Returns `false` if CHOLMOD reports that a downdate lost positive definiteness; the factor is
then unusable and must be rebuilt by the caller. In practice `cholmod_l_updown` does not check
definiteness and returns success with a negative pivot, so callers must test the pivots
themselves (`min_scaled_pivot`); the solver does so after every downdate.
"""
function updown!(F::LDLtFactor, update::Bool, rows::AbstractVector{<:Integer},
                 vals::AbstractVector{Float64}, nnz::Integer)
    _check_open(F)
    nnz <= F.n || throw(ArgumentError("updown!: nnz = $nnz exceeds dimension $(F.n)"))
    nnz == 0 && return true                          # v = 0: nothing to do
    # CHOLMOD needs the pattern of C sorted: sort the positions by row index
    perm = sortperm(view(rows, 1:nnz))
    C_i, C_x = F.C_i, F.C_x
    @inbounds for t in 1:nnz
        C_i[t] = rows[perm[t]] - 1                   # 0-based row index
        C_x[t] = vals[perm[t]]
    end
    F.C_p[2] = nnz
    cp = common_ptr(F.common)
    flag = update ? CHOLMOD_TRUE : CHOLMOD_FALSE
    ret = GC.@preserve F cholmod_l_updown(flag, F.C, F.L, cp)
    st = status(F.common)
    # CHOLMOD_NOT_POSDEF would be a warning, not a failure (in practice updown does not check)
    ret == CHOLMOD_TRUE || st == CHOLMOD_NOT_POSDEF || throw(CholmodError("cholmod_l_updown", st))
    return ret == CHOLMOD_TRUE && st != CHOLMOD_NOT_POSDEF
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
    # Bset = pattern of the right-hand side; CHOLMOD then solves only on reach_L(Bset)
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
        # D[k] is the first stored entry of column k of the simplicial factor
        d = unsafe_load(f.x, unsafe_load(f.p, k) + 1)
        m = min(m, d / max(scale[k], floatmin(Float64)))
    end
    return m
end

"""
    factor_stats(F, active) -> NamedTuple

Contents of the factor `L` (strictly lower part; the diagonal holds `D` and is not counted):

- `allocated`: entries CHOLMOD has reserved for `L` (`nzmax`, including the diagonal)
- `stored`: off-diagonal entries currently stored in the pattern of `L`. `rowdel` and
  rank-one updates zero entries numerically but keep them in the pattern, so this is what
  every solve and update actually pays for
- `effective`: stored off-diagonal entries that are numerically nonzero
- `active_stored`, `active_effective`: the same, restricted to the columns in `active`
- `memory`, `memory_peak`: bytes held by CHOLMOD now / at most since the factor was created

One pass over the stored entries, `O(stored + n)`.
"""
function factor_stats(F::LDLtFactor, active::AbstractVector{<:Integer})
    _check_open(F)
    f = unsafe_load(F.L)
    is_active = falses(F.n)
    for k in active
        is_active[k] = true
    end
    stored = effective = astored = aeffective = 0
    @inbounds for j in 1:F.n
        start = unsafe_load(f.p, j)           # 0-based start of column j
        cnt = unsafe_load(f.nz, j)            # stored entries of column j, diagonal first
        # off-diagonal entries: positions start+1 … start+cnt-1
        nzj = 0
        for t in (start + 1):(start + cnt - 1)
            unsafe_load(f.x, t + 1) != 0.0 && (nzj += 1)
        end
        stored += cnt - 1
        effective += nzj
        if is_active[j]
            astored += cnt - 1
            aeffective += nzj
        end
    end
    return (allocated = Int(f.nzmax), stored = stored, effective = effective,
            active_stored = astored, active_effective = aeffective,
            memory = memory_inuse(F.common), memory_peak = memory_peak(F.common))
end

"""
    last_reach(F) -> Int

Size of `reach_L(Bset)` of the last sparse right-hand-side solve, i.e. the number of rows
of the factor that solve touched (0 if no sparse solve has happened yet).
"""
function last_reach(F::LDLtFactor)
    _check_open(F)
    F.Xset[] == C_NULL && return 0
    Xh = unsafe_load(F.Xset[])
    # Xset is an n × 1 pattern: its column pointer p[1] = 0, p[2] = number of entries
    return Int(unsafe_load(Xh.p, 2))
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
        cholmod_l_free_sparse(Ref(F.C), cp)
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
