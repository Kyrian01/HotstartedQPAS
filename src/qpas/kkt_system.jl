# KKT system of the regularized master problem in nullspace form (thesis §4.3–4.4).
#
# The KKT system (thesis eq. 4.17) is never formed. With p = Zy it reduces to
#
#     H_eff y = -δ Zᵀg̃,     H_eff = (A_eff Z)ᵀ(A_eff Z),
#
# where A_eff holds the equality rows and the inequality rows whose slack is in the working set
# (docs/qpas_inequality_extension.md, Cor. 3.3). Without inequality rows A_eff = A and H_eff is
# the reduced Hessian H of the thesis. H_eff is kept as an LDLᵀ factor of a matrix of size n_max
# that is the identity on indices outside N, and updated in place whenever N or the slack status
# changes. The factor is either CHOLMOD's (stores all n_max indices, sparse) or the dense backend
# (stores only N, |N| ≤ m), see src/backends/ldlt_interface.jl:
#
#   index j enters N  →  rowadd(j, H_eff[:, j])         (reduction, thesis §4.4.2)
#   index j leaves N  →  rowdel(j)                      (expansion, thesis §4.4.1)
#   representative of block b changes  →  rowdel + rowadd for every j ∈ N of block b
#   slack of row i becomes active      →  H_eff += vvᵀ  (row i enters A_eff)
#   slack of row i becomes free        →  H_eff -= vvᵀ  (row i leaves A_eff),  v = Zᵀ(row i of A)
#
# Column j of H_eff is assembled on demand from the column pool:
#   u = a_j - a_rep(j) with the free rows zeroed,  H_eff[k, j] = (a_k - a_rep(k))ᵀ u  (k ∈ N),
# which costs O((|N| + p) · nnz per column), independent of the total number of columns.

"""
    KKTSystem(n_blocks, n_rows, capacity, surplus; debug = false, factorization = :cholmod)

Linear-algebra state of the nullspace KKT system:

- `ordering::ColumnOrdering`: the partition [S, W, N] of the columns; it fully determines the
  nullspace basis `Z` and `AZ` (see `nullspace.jl`), which are therefore implicit
- `surplus::SurplusSet`: the slacks of the inequality rows and the split `W_s` / `N_s`; it
  determines the row mask of `H_eff`
- `factor::F`: LDLᵀ of the effective reduced Hessian (`Cholmod.LDLtFactor` or
  `DenseLDLt.DenseLDLtFactor`, both `<: UpdatableLDLt`)
- `hdiag`: unmasked `‖a_j - a_rep(j)‖²` at activation, the scale for pivot/singularity tests
- `updates_since_refactor`, `total_updates`: factor row additions/deletions + rank-one updates
- `timer`: time and call count of every factor operation (`FACTOR_TIMER_SECTIONS`)
- scratch buffers for Hessian column and rank-one vector assembly
- `debug_H`: explicit mirror of `H_eff` (active entries only, `(k, j) => H_eff[k, j]`),
  kept only in debug mode
"""
mutable struct KKTSystem{F<:UpdatableLDLt}
    ordering::ColumnOrdering
    surplus::SurplusSet
    factor::F
    hdiag::Vector{Float64}
    u::Vector{Float64}                 # dense n_rows scatter of a_j - a_rep(j); zero between uses
    rep_dot::Vector{Float64}           # (a_rep(b))ᵀ u per block
    rep_dot_valid::Vector{Bool}
    col_rows::Vector{Int}              # assembled Hessian column (indices, values)
    col_vals::Vector{Float64}
    v_rows::Vector{Int}                # assembled rank-one vector v (indices, values)
    v_vals::Vector{Float64}
    affected::Vector{Int}              # ν: N-columns of a block whose representative changes
    updates_since_refactor::Int
    total_updates::Int
    timer::SectionTimer
    debug_H::Union{Nothing,Dict{Tuple{Int,Int},Float64}}
end

function KKTSystem(n_blocks::Integer, n_rows::Integer, capacity::Integer,
                   surplus::SurplusSet = SurplusSet(Int[], Float64[], n_rows); debug::Bool = false,
                   factorization::Symbol = :cholmod)
    factor = make_factor(factorization, capacity, n_rows)
    return KKTSystem(ColumnOrdering(n_blocks, capacity), surplus, factor,
                     zeros(capacity), zeros(n_rows), zeros(n_blocks), fill(false, n_blocks),
                     zeros(Int, capacity), zeros(capacity), zeros(Int, capacity), zeros(capacity),
                     Int[], 0, 0, SectionTimer(FACTOR_TIMER_SECTIONS),
                     debug ? Dict{Tuple{Int,Int},Float64}() : nothing)
end

"""
    make_factor(factorization, capacity, n_rows) -> UpdatableLDLt

Identity factor of size `capacity` (= `n_max`) for the chosen backend. The dense backend
starts with `n_rows + 1` slots: `|N| ≤ m` while `H_eff` is positive definite, plus one slot for
a dependent column that is added and then undone (it grows if ever needed).
"""
function make_factor(factorization::Symbol, capacity::Integer, n_rows::Integer)
    factorization === :cholmod && return Cholmod.LDLtFactor(capacity)
    factorization === :dense &&
        return DenseLDLt.DenseLDLtFactor(capacity; capacity = min(capacity, n_rows + 1))
    throw(ArgumentError("unknown factorization :$factorization (allowed: $(join(FACTORIZATIONS, ", ")))"))
end

"""
    hessian_column!(kkt, pool, j) -> (nnz, hjj, scale)

Assemble column `j` of the effective reduced Hessian against the current set `N` (which must
not contain `j`) into `kkt.col_rows[1:nnz]`, `kkt.col_vals[1:nnz]`. The last entry is the
diagonal `hjj = ‖a_j - a_rep(j)‖²` over the rows of `A_eff`; `scale` is the same norm over all
rows (the pivot scale, which does not change when rows are masked).
"""
function hessian_column!(kkt::KKTSystem, pool::ColumnPool, j::Int)
    return @timed_section kkt.timer :assemble _hessian_column!(kkt, pool, j)
end

function _hessian_column!(kkt::KKTSystem, pool::ColumnPool, j::Int)
    ord = kkt.ordering
    block = pool.block
    u = kkt.u
    r = representative(ord, block, j)

    # dense scatter of u = a_j - a_rep(j) = A z_j
    column_axpy!(u, 1.0, pool, j)
    column_axpy!(u, -1.0, pool, r)
    # unmasked ‖u‖², used as the pivot scale of index j
    scale = column_dot(pool, j, u) - column_dot(pool, r, u)
    # rows with a free slack are not part of A_eff: drop them from u (Cor. 3.3)
    masked = any_free(kkt.surplus)
    masked && _mask_rows!(u, kkt.surplus.row_free, pool, j, r)
    fill!(kkt.rep_dot_valid, false)

    nnz = 0
    rows, vals = kkt.col_rows, kkt.col_vals
    @inbounds for k in items(ord.N)
        b = block[k]
        # (a_rep(b))ᵀu is shared by every N-column of block b: compute it once
        if !kkt.rep_dot_valid[b]
            kkt.rep_dot[b] = column_dot(pool, ord.rep[b], u)
            kkt.rep_dot_valid[b] = true
        end
        # H_eff[k, j] = (a_k - a_rep(k))ᵀ u
        h = column_dot(pool, k, u) - kkt.rep_dot[b]
        if h != 0.0
            nnz += 1
            rows[nnz] = k
            vals[nnz] = h
        end
    end
    # diagonal entry ‖u‖² over the rows of A_eff, stored last
    hjj = masked ? column_dot(pool, j, u) - column_dot(pool, r, u) : scale
    nnz += 1
    rows[nnz] = j
    vals[nnz] = hjj

    # reset the scatter buffer on the (only) rows it touched
    @inbounds for i in column_rows(pool, j)
        u[i] = 0.0
    end
    @inbounds for i in column_rows(pool, r)
        u[i] = 0.0
    end
    return nnz, hjj, scale
end

# Zero the entries of the scatter `u = a_j - a_r` on rows whose slack is free.
@inline function _mask_rows!(u, row_free, pool, j, r)
    @inbounds for i in column_rows(pool, j)
        row_free[i] && (u[i] = 0.0)
    end
    @inbounds for i in column_rows(pool, r)
        row_free[i] && (u[i] = 0.0)
    end
    return nothing
end

"""
    insert_column!(kkt, j, nnz, scale)

Activate index `j` in the factor with the column previously assembled by
[`hessian_column!`](@ref) and move `j` into `N`.
"""
function insert_column!(kkt::KKTSystem, j::Int, nnz::Int, scale::Float64)
    # row/column j of the factor goes from e_j to the assembled Hessian column
    @timed_section kkt.timer :rowadd LDLt.rowadd!(kkt.factor, j, kkt.col_rows, kkt.col_vals, nnz)
    kkt.hdiag[j] = scale
    kkt.updates_since_refactor += 1
    kkt.total_updates += 1
    if kkt.debug_H !== nothing
        H = kkt.debug_H
        for t in 1:nnz
            k = kkt.col_rows[t]
            H[(k, j)] = kkt.col_vals[t]
            H[(j, k)] = kkt.col_vals[t]
        end
    end
    move_to_N!(kkt.ordering, j)
    return nothing
end

"Assemble column `j` and activate it (no singularity check)."
function activate!(kkt::KKTSystem, pool::ColumnPool, j::Int)
    nnz, _, scale = hessian_column!(kkt, pool, j)
    insert_column!(kkt, j, nnz, scale)
    return nothing
end

"""
    deactivate!(kkt, j)

Reset row/column `j` of the factor to the identity and remove `j` from `N`. The caller
decides where `j` goes next (W or S).
"""
function deactivate!(kkt::KKTSystem, j::Int)
    @timed_section kkt.timer :rowdel LDLt.rowdel!(kkt.factor, j)
    kkt.updates_since_refactor += 1
    kkt.total_updates += 1
    if kkt.debug_H !== nothing
        for k in items(kkt.ordering.N)
            delete!(kkt.debug_H, (k, j))
            delete!(kkt.debug_H, (j, k))
        end
    end
    delete!(kkt.ordering.N, j)
    return nothing
end

"""
    surplus_vector!(kkt, pool, σ) -> nnz

Assemble the rank-one vector of slack `σ` into `kkt.v_rows[1:nnz]`, `kkt.v_vals[1:nnz]`:

    v = Zᵀ âᵢ,   v_k = (a_k)ᵢ - (a_rep(k))ᵢ   (k ∈ N),   i = row of σ,

i.e. row `i` of `AZ`. Activating the slack adds `vvᵀ` to `H_eff`, freeing it subtracts `vvᵀ`
(docs/qpas_inequality_extension.md, Thm. 4.1/4.2).
"""
function surplus_vector!(kkt::KKTSystem, pool::ColumnPool, σ::Int)
    return @timed_section kkt.timer :assemble _surplus_vector!(kkt, pool, σ)
end

function _surplus_vector!(kkt::KKTSystem, pool::ColumnPool, σ::Int)
    ord = kkt.ordering
    block = pool.block
    i = kkt.surplus.row[σ]
    fill!(kkt.rep_dot_valid, false)
    nnz = 0
    @inbounds for k in items(ord.N)
        b = block[k]
        # entry (a_rep(b))ᵢ, cached per block
        if !kkt.rep_dot_valid[b]
            kkt.rep_dot[b] = column_entry(pool, ord.rep[b], i)
            kkt.rep_dot_valid[b] = true
        end
        vk = column_entry(pool, k, i) - kkt.rep_dot[b]
        if vk != 0.0
            nnz += 1
            kkt.v_rows[nnz] = k
            kkt.v_vals[nnz] = vk
        end
    end
    return nnz
end

"""
    rank_one_update!(kkt, update, nnz) -> Bool

`H_eff ± vvᵀ` with the vector assembled by [`surplus_vector!`](@ref) (`+` if `update`).
Returns `false` if the backend flags a downdate as not positive definite.
"""
function rank_one_update!(kkt::KKTSystem, update::Bool, nnz::Int)
    ok = @timed_section kkt.timer :updown LDLt.updown!(kkt.factor, update, kkt.v_rows,
                                                         kkt.v_vals, nnz)
    # a rank-one change leaves stored entries behind like rowdel: count it for refactoring
    kkt.updates_since_refactor += 1
    kkt.total_updates += 1
    if kkt.debug_H !== nothing
        sgn = update ? 1.0 : -1.0
        H = kkt.debug_H
        for s in 1:nnz, t in 1:nnz
            k, l = kkt.v_rows[s], kkt.v_rows[t]
            H[(k, l)] = get(H, (k, l), 0.0) + sgn * kkt.v_vals[s] * kkt.v_vals[t]
        end
    end
    return ok
end

"""
    min_pivot_ratio(kkt) -> Float64

Smallest `D[k] / ‖a_k - a_rep(k)‖²` over `k ∈ N` (`Inf` if `N` is empty). Values near zero
indicate a (numerically) singular effective reduced Hessian. The unmasked norm is used as scale
so that a column whose masked part vanishes is still recognised as singular.
"""
min_pivot_ratio(kkt::KKTSystem) =
    LDLt.min_scaled_pivot(kkt.factor, items(kkt.ordering.N), kkt.hdiag)

"""
    refactor!(kkt, pool)

Rebuild the factorization from scratch: reset to the identity and re-activate every index
of `N` against the current row mask. Used for periodic refactorization and as a fallback when
pivots degrade.

Indices are re-activated in increasing order: with CHOLMOD's natural ordering every
`rowadd` then appends the last row of the factor (one triangular solve, no update of
trailing columns). The dense backend always appends, in any order.
"""
function refactor!(kkt::KKTSystem, pool::ColumnPool)
    LDLt.reset!(kkt.factor)
    ord = kkt.ordering
    order = sort!(copy(items(ord.N)))
    empty!(ord.N)
    kkt.debug_H === nothing || empty!(kkt.debug_H)
    for j in order
        activate!(kkt, pool, j)
    end
    kkt.updates_since_refactor = 0
    return nothing
end

"""
    verify_kkt(kkt, pool; tol = 1e-8) -> Float64

Debug check, run after every QPAS iteration in debug mode:

1. the ordering invariants hold;
2. the effective reduced Hessian on `N`, recomputed from scratch as `(A_eff Z)ᵀ(A_eff Z)`,
   equals the incrementally maintained mirror `debug_H` (if present);
3. the factor reproduces it: `(L D Lᵀ)[N, N] = H_eff`, and every inactive index has
   `D[k] = 1` and no coupling to `N` in `L`.

All work is restricted to `N` (`|N| ≤ m`), so this stays usable on full-size instances.
Throws on failure; returns the relative factorization error.
"""
function verify_kkt(kkt::KKTSystem, pool::ColumnPool; tol::Real = 1e-8)
    ord = kkt.ordering
    check_ordering(ord, pool, pool.n)
    idx, H = reduced_hessian(ord, pool; row_free = kkt.surplus.row_free)
    scale = max(1.0, maximum(abs, H; init = 0.0))

    if kkt.debug_H !== nothing
        pos = Dict(j => t for (t, j) in enumerate(idx))
        mirror = zeros(length(idx), length(idx))
        for ((k, j), v) in kkt.debug_H
            (haskey(pos, k) && haskey(pos, j)) ||
                error("KKT verification: mirror holds entry ($k, $j) outside N")
            mirror[pos[k], pos[j]] = v
        end
        err = maximum(abs, mirror - H; init = 0.0) / scale
        err <= tol || error("KKT verification: tracked reduced Hessian is off by $err")
    end

    # LDLᵀ restricted to the rows of N must reproduce H_eff
    L, d = LDLt.factor_matrices(kkt.factor)
    err = _factor_error(L, d, idx, H, scale)
    err <= tol || error("KKT verification: LDLᵀ factor is off by $err on N")
    # inactive indices must still be identity rows, decoupled from N
    active = falses(length(d))
    active[idx] .= true
    inactive = findall(!, active)
    maximum(abs, d[inactive] .- 1.0; init = 0.0) <= tol ||
        error("KKT verification: an inactive pivot differs from 1")
    maximum(abs, L[idx, inactive]; init = 0.0) <= tol * scale ||
        error("KKT verification: L couples N to inactive indices")
    return err
end

_factor_error(L, d, idx, H, scale) =
    (LN = L[idx, :]; maximum(abs, Matrix(LN * Diagonal(d) * LN') - H; init = 0.0) / scale)

"""
    factor_residual(kkt, pool) -> Float64

`max |(L D Lᵀ)[N, N] - H_eff| / max(1, max |H_eff|)` with `H_eff` rebuilt from the pool: how
far the incrementally updated factor has drifted from the matrix it represents. Used by the
diagnostics traces; `O(m|N|² + |N|³)`.
"""
function factor_residual(kkt::KKTSystem, pool::ColumnPool)
    idx, H = reduced_hessian(kkt.ordering, pool; row_free = kkt.surplus.row_free)
    isempty(idx) && return 0.0
    L, d = LDLt.factor_matrices(kkt.factor)
    return _factor_error(L, d, idx, H, max(1.0, maximum(abs, H)))
end
